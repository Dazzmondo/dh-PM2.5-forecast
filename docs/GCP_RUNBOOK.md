# Runbook: the compute benchmark on a Google Cloud VM

This is the exact sequence for running the Milestone 3 compute benchmark (`src/benchmark.py`, experiment E11) on a GPU virtual machine,
and for stopping everything afterwards. One VM with one NVIDIA T4 GPU runs all three tests, so the CPU and GPU results come from the
same machine and the same image:

| Test | What it measures |
|---|---|
| `xgb_threads` | XGBoost fit time on 1, 2, 4, 8 CPU threads (speed-up, efficiency, Amdahl's law) |
| `xgb_device` | XGBoost on the CPU and on the GPU (`device="cuda"`) |
| `gru_device` | The GRU for a fixed number of epochs on the CPU and on the GPU |

Everything below that needs a project, zone, bucket or price is written as a `<PLACEHOLDER>`. Nothing in this runbook contains a
credential, a project ID or a bucket name; none belongs in the repository.

## 0. Values to fill in

| Placeholder | Meaning |
|---|---|
| `<PROJECT_ID>` | the GCP project that holds the bucket and the credits |
| `<BUCKET>` | the bucket with the processed data (`GCS_BUCKET_NAME`) |
| `<REGION>` / `<ZONE>` | an EU region and a zone in it that offers T4 GPUs (step 2) |
| `<COMMIT>` | the pushed commit to benchmark (the benchmark records it) |
| `<DATASET_RUN_ID>` | the dataset run to link to; currently `20261019T213934Z` |
| `<RATE_CPU>`, `<RATE_GPU>` | the machine's price per hour without and with the GPU, from the current price list (note the date) |

## 1. Before creating anything

1. **Quota.** Console → IAM & Admin → Quotas. Filter for **GPUs (all regions)** and for **NVIDIA T4 GPUs** in `<REGION>`. If either limit
   is 0, request an increase to 1. A new project often starts at 0, and approval can take time, so request it first.
2. **Credits.** Console → Billing → Credits: check the expiry date and the remaining balance, so the session happens before they run out
   and fits inside the balance.
3. **Budget alert.** Console → Billing → Budgets & alerts → Create budget, with alerts at a small amount. This is the safety net if a VM
   is left running.
4. **Code.** Commit and push everything the benchmark needs. The VM clones the repository, so only pushed code reaches it; `<COMMIT>` is the
   commit you pushed.
5. **A quiet machine.** Nothing else may run on the VM during a timed test. Start the VM only when ready, and stop it as soon as the
   results are copied back (step 11).

## 2. Find a zone with T4 GPUs

```
gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4 AND zone~europe" --project=<PROJECT_ID>
```

Pick a zone from the list. If creating the VM later fails with a resource-availability error, try another zone in the list.

## 3. A read-only identity for the VM (least privilege)

The benchmark only reads the processed data, so the VM gets its own service account that can read the bucket and nothing else:

```
gcloud iam service-accounts create pm25-benchmark --display-name="PM2.5 benchmark (read-only)" --project=<PROJECT_ID>
gcloud storage buckets add-iam-policy-binding gs://<BUCKET> --member="serviceAccount:pm25-benchmark@<PROJECT_ID>.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
```

## 4. Create the VM

```
gcloud compute instances create pm25-bench-gpu --project=<PROJECT_ID> --zone=<ZONE> --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced --service-account=pm25-benchmark@<PROJECT_ID>.iam.gserviceaccount.com --scopes=storage-ro
```

- A GPU VM must use `TERMINATE` on host maintenance. If `gcloud` reports the flag as deprecated, `--on-host-maintenance=TERMINATE` is the newer spelling.
- `n1-standard-8` has 8 vCPUs; on Google Cloud a vCPU is a hyperthread, so these are 4 physical cores. The boot disk is large because the GPU image holds the CUDA build of PyTorch.
- It is an on-demand VM, not a Spot VM: a timing run must not be pre-empted.

Connect to it (the rest of the steps run on the VM unless a step says "from the laptop"):

```
gcloud compute ssh pm25-bench-gpu --zone=<ZONE> --project=<PROJECT_ID>
```

## 5. Install the NVIDIA driver

```
sudo apt-get update && sudo apt-get install -y git python3 curl ca-certificates
curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver
```

The installation usually restarts the VM. If your SSH session drops, reconnect and run the same `install_driver` command again until it
finishes. Then check:

```
nvidia-smi
```

It must show a Tesla T4. Note the driver and CUDA versions for the report. (Google's installer replaces the older
`install_gpu_driver.py` script, which is deprecated.)

## 6. Install Docker and the NVIDIA Container Toolkit

```
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

Log out and back in so the group change applies. Then install the NVIDIA Container Toolkit with the apt steps in NVIDIA's installation
guide (the repository setup and package versions change over time, so follow the guide, not a copied list), and connect it to Docker:

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The last command must print the same T4 table from inside a container.

## 7. Get the code and the data

```
git clone https://github.com/Dazzmondo/dh-PM2.5-forecast.git
cd dh-PM2.5-forecast
git checkout <COMMIT>
git rev-parse HEAD
mkdir -p processed_data/features processed_data/scalers
gcloud storage cp gs://<BUCKET>/processed/features/pm25_features.parquet processed_data/features/pm25_features_with_splits.parquet
gcloud storage cp gs://<BUCKET>/processed/three_hour/three_hour_aggregated.parquet processed_data/three_hour_aggregated.parquet
gcloud storage cp gs://<BUCKET>/processed/scalers/scalers.json processed_data/scalers/scalers.json
```

The features object in the bucket has a different name from the local file; the bytes are identical. Check all three against the dataset
record (`20261019T213934Z`):

```
sha256sum processed_data/features/pm25_features_with_splits.parquet processed_data/three_hour_aggregated.parquet processed_data/scalers/scalers.json
```

| File | Expected SHA-256 |
|---|---|
| `pm25_features_with_splits.parquet` | `b2cd1f6ecfdb5d313688538dc415c3768aa987e0002f2b559b4d05895695ffc4` |
| `three_hour_aggregated.parquet` | `f3d9209ded925b93c10c60ae9ea740cd031e3ea56c868abd465e607d8e9d45a3` |
| `scalers.json` | `0bde1d04c253300c27ea9d47f136fae00f6c098a3231181bf93ad9830f7588c8` |

If a hash differs, stop: the data is not the dataset the experiments used.

## 8. Build the image

```
docker build -t pm25-m3:gpu --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg TORCH_INDEX=https://pypi.org/simple .
IMAGE_ID=$(docker image inspect pm25-m3:gpu --format '{{.Id}}')
echo $IMAGE_ID
```

- The `TORCH_INDEX` build argument selects the CUDA build of PyTorch (the default is the CPU build). Expect a long build and a large
  image. Build before the timed tests, and note the time the VM has been running.
- The image ID is recorded on every run as `image_digest`. A registry digest exists only after pushing to a registry (for example
  Artifact Registry); the image ID is what this build can record.
- The Linux XGBoost wheel is expected to include CUDA support. The benchmark verifies it: it asks XGBoost which device it really used
  and stops if a `cuda` run silently fell back to the CPU, so a CPU run can never be recorded as a GPU run.

## 9. Check first, then run

All runs use the same options. Define them once in the shell:

```
RUN="docker run --rm --gpus all -v $PWD/processed_data:/app/processed_data:ro -v $PWD/mlruns:/app/mlruns -v $PWD/reports:/app/reports -e DATASET_RUN_ID=<DATASET_RUN_ID> -e IMAGE_DIGEST=$IMAGE_ID pm25-m3:gpu"
```

**9a. A few-minute check (its own MLflow experiment, not a result):**

```
$RUN src/benchmark.py --config configs/compute_benchmark_check.yaml --machine-type n1-standard-8-t4 --region <REGION> --physical-cores 4
```

It must finish without a SKIPPED line for `cuda`, and the summary must list `cuda` rows for `xgb_device` and `gru_device`. If a `cuda`
test is skipped, the GPU is not visible to the container; go back to step 6.

**9b. The CPU tests** (nothing else running on the VM; the rate is the machine without the GPU):

```
$RUN src/benchmark.py --config configs/compute_benchmark.yaml --only xgb_threads --machine-type n1-standard-8 --region <REGION> --physical-cores 4 --hourly-rate <RATE_CPU>
```

**9c. The GPU tests** (the rate is the machine with the GPU):

```
$RUN src/benchmark.py --config configs/compute_benchmark.yaml --only xgb_device gru_device --machine-type n1-standard-8-t4 --region <REGION> --physical-cores 4 --hourly-rate <RATE_GPU>
```

In 9c the CPU rows of `xgb_device` and `gru_device` are priced at the GPU machine's rate by the script's cost column. For the report, work
out each row's cost as median fit seconds × rate ÷ 3600 using `<RATE_CPU>` for CPU rows and `<RATE_GPU>` for GPU rows.

## 10. Record what the report needs

Run these on the VM and copy the output into your notes:

```
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
git rev-parse HEAD
echo $IMAGE_ID
```

| Item | From |
|---|---|
| Machine type, zone, region, date | step 4 |
| CPU model, logical and physical cores | `lscpu` |
| GPU model, driver and CUDA versions | `nvidia-smi` |
| Commit and image ID | steps 7 and 8 |
| Hourly rates (and the date of the price list) | the pricing page for the machine type and the T4 |
| Time the VM was running, and the approximate cost | VM start and stop times × `<RATE_GPU>` |

Export the run tables on the VM, then copy the results back **from the laptop**:

```
$RUN src/export_runs.py
```

```
gcloud compute scp --recurse pm25-bench-gpu:~/dh-PM2.5-forecast/reports/m3 ./reports/m3_gcp --zone=<ZONE> --project=<PROJECT_ID>
gcloud compute scp --recurse pm25-bench-gpu:~/dh-PM2.5-forecast/mlruns/artifacts ./mlruns_gcp_artifacts --zone=<ZONE> --project=<PROJECT_ID>
```

The first copy holds the exported run summary; the second holds the per-repeat and summary CSV files of each benchmark run. (The VM's MLflow database records its container
paths, so view the results from these files, not by opening the VM's database on the laptop.)

## 11. Stop and delete everything

As soon as the results are on the laptop, from the laptop:

```
gcloud compute instances delete pm25-bench-gpu --zone=<ZONE> --project=<PROJECT_ID> --delete-disks=all
gcloud compute instances list --project=<PROJECT_ID>
gcloud compute disks list --project=<PROJECT_ID>
```

Both lists must be empty of benchmark resources. A stopped VM still pays for its disk, so delete rather than stop unless a rerun is
certain. Then check Billing → Reports the next day for the actual charge, and write it in the report beside the estimate.

## Troubleshooting

| Symptom | Likely cause and action |
|---|---|
| `gcloud` reports quota exceeded for `NVIDIA_T4_GPUS` or `GPUS_ALL_REGIONS` | Step 1: the limit is still 0; wait for the increase |
| Creating the VM fails with a resource-availability or "zone does not have enough resources" error | The zone has no free T4 capacity right now; try another zone from step 2 |
| `nvidia-smi` not found after step 5 | The installer was interrupted by the restart; reconnect and run `install_driver` again |
| `docker: Error response ... could not select device driver "nvidia"` | The container toolkit is not connected to Docker; repeat the `nvidia-ctk runtime configure` step and restart Docker |
| The benchmark stops with "XGBoost was asked for device='cuda' but ran on 'cpu'" | XGBoost cannot see the GPU in the container; check `docker run --gpus all ... nvidia-smi` and that the image was built with the GPU `TORCH_INDEX` |
| A hash in step 7 does not match | The files are not the dataset the experiments used; stop and check the bucket's `manifests/dataset_versions.jsonl` |
| `permission denied` reading the bucket | The VM's service account lacks `roles/storage.objectViewer` on the bucket (step 3) |

## Sources

- [Install GPU drivers (Compute Engine)](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu)
- [Create an N1 VM that has attached GPUs](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose)
- [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
