# Runbook: the compute comparison on two Google Cloud VMs (and the laptop)

This is the exact sequence for running the Milestone 3 compute benchmark (`src/benchmark.py`, experiment E11), the GRU profile
(`src/profile_gru.py`) and the reproduction check (`src/train.py`) on two virtual machines, and for stopping everything afterwards. The
two VMs have the same machine type, image and operating system; the only difference is the GPU, so any difference between them is the GPU.
The laptop is the third environment.

| Environment (`RUN_ENVIRONMENT`) | Machine | GPU |
|---|---|---|
| `laptop` | the development laptop (4 physical cores, 8 threads, 15.8 GB) | none |
| `gcp-cpu-vm` | `pm25-bench-cpu`: n1-standard-8 | none |
| `gcp-gpu-vm` | `pm25-bench-gpu`: n1-standard-8 plus one NVIDIA T4 | T4 |

What runs where (the same configuration files everywhere; the tuned XGBoost and the tuned GRU of `docs/M3.md` sections 3.6 and 3.7):

| Test | laptop | CPU VM | GPU VM |
|---|---|---|---|
| `xgb_threads`: fit time on 1, 2, 4, 8 CPU threads (speed-up, efficiency, Amdahl's law) | yes | yes | no (the CPU is the same as the CPU VM's) |
| `xgb_device`: XGBoost on the CPU and on the GPU (`device="cuda"`) | CPU only | CPU only | CPU (a same-machine control) and GPU |
| `gru_device`: the GRU for a fixed number of epochs on the CPU and on the GPU | CPU only | CPU only | CPU and GPU |
| GRU profile (`profile_gru.py`) | `--device cpu` | `--device cpu` | `--device cuda` |
| Reproduction of the tuned XGBoost (`train.py`) | in the container | yes | CPU config and GPU config |

Everything below that needs a project, zone, bucket or price is written as a `<PLACEHOLDER>`. Nothing in this runbook contains a
credential, a project ID or a bucket name; none belongs in the repository.

## 0. Values to fill in

| Placeholder | Meaning |
|---|---|
| `<PROJECT_ID>` | the GCP project that holds the bucket and the credits |
| `<BUCKET>` | the bucket with the processed data (`GCS_BUCKET_NAME`) |
| `<REGION>` / `<ZONE>` | an EU region and a zone in it that offers T4 GPUs (step 2); both VMs use the same zone |
| `<COMMIT>` | the pushed commit to run (every run records it) |
| `<DATASET_RUN_ID>` | the dataset run to link to; currently `20261019T213934Z` |
| `<RATE_CPU>`, `<RATE_GPU>` | the price per hour of the CPU VM and of the GPU VM (machine plus T4), from the current price list (note the date) |

## 1. Before creating anything

1. **Quota.** Console → IAM & Admin → Quotas. For the GPU VM, filter for **GPUs (all regions)** and for **NVIDIA T4 GPUs** in
   `<REGION>`; if either limit is 0, request an increase to 1. Both VMs also need enough N1 vCPUs in the region (8 each). A new project often
   starts with a GPU limit of 0 and approval can take time, so request it first.
2. **Credits.** Console → Billing → Credits: check the expiry date and the remaining balance, so the session happens before they run out
   and fits inside the balance.
3. **Budget alert.** Console → Billing → Budgets & alerts → Create budget, with alerts at a small amount. This is the safety net if a VM
   is left running.
4. **Code.** Commit and push everything the runs need. The VMs clone the repository, so only pushed code reaches them; `<COMMIT>` is the
   commit you pushed.
5. **A quiet machine.** Nothing else may run on a VM during a timed test. Start a VM only when ready, and delete it as soon as its results are
   copied back (step 11). The CPU VM and the GPU VM may run at the same time (they are separate machines); never run two timed tests on one VM.

## 2. Find a zone with T4 GPUs

```
gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4 AND zone~europe" --project=<PROJECT_ID>
```

Pick a zone from the list and use it for both VMs. If creating the GPU VM later fails with a resource-availability error, try another zone in
the list (and create the CPU VM in the same one).

## 3. A read-only identity for the VMs (least privilege)

The runs only read the processed data, so the VMs get their own service account that can read the bucket and nothing else:

```
gcloud iam service-accounts create pm25-benchmark --display-name="PM2.5 benchmark (read-only)" --project=<PROJECT_ID>
gcloud storage buckets add-iam-policy-binding gs://<BUCKET> --member="serviceAccount:pm25-benchmark@<PROJECT_ID>.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
```

## 4. Create the VMs

The CPU VM:

```
gcloud compute instances create pm25-bench-cpu --project=<PROJECT_ID> --zone=<ZONE> --machine-type=n1-standard-8 --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=50GB --boot-disk-type=pd-balanced --service-account=pm25-benchmark@<PROJECT_ID>.iam.gserviceaccount.com --scopes=storage-ro --labels=project=pm25-m3,purpose=benchmark
```

The GPU VM:

```
gcloud compute instances create pm25-bench-gpu --project=<PROJECT_ID> --zone=<ZONE> --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced --service-account=pm25-benchmark@<PROJECT_ID>.iam.gserviceaccount.com --scopes=storage-ro --labels=project=pm25-m3,purpose=benchmark
```

- A GPU VM must use `TERMINATE` on host maintenance. If `gcloud` reports the flag as deprecated, `--on-host-maintenance=TERMINATE` is the newer spelling.
- `n1-standard-8` has 8 vCPUs; on Google Cloud a vCPU is a hyperthread, so these are 4 physical cores, the same thread count as the
  laptop. The GPU VM's disk is larger because the GPU image holds the CUDA build of PyTorch.
- Both are on-demand VMs, not Spot VMs: a timing run must not be pre-empted. The labels let the charge be found in Billing reports.

Connect to a VM (the rest of the steps run on it unless a step says "from the laptop"):

```
gcloud compute ssh pm25-bench-cpu --zone=<ZONE> --project=<PROJECT_ID>
gcloud compute ssh pm25-bench-gpu --zone=<ZONE> --project=<PROJECT_ID>
```

On both VMs install the basics and Docker:

```
sudo apt-get update && sudo apt-get install -y git python3 curl ca-certificates
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

Log out and back in so the group change applies.

## 5. GPU VM only: install the NVIDIA driver

```
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

## 6. GPU VM only: the NVIDIA Container Toolkit

Install the NVIDIA Container Toolkit with the apt steps in NVIDIA's installation guide (the repository setup and package versions change
over time, so follow the guide, not a copied list), and connect it to Docker:

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The last command must print the same T4 table from inside a container.

## 7. Get the code and the data (both VMs)

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

On the CPU VM (the default build uses the CPU build of PyTorch):

```
docker build -t pm25-m3:cpu --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
IMAGE_ID=$(docker image inspect pm25-m3:cpu --format '{{.Id}}')
echo $IMAGE_ID
```

On the GPU VM (the `TORCH_INDEX` build argument selects the CUDA build of PyTorch):

```
docker build -t pm25-m3:gpu --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg TORCH_INDEX=https://pypi.org/simple .
IMAGE_ID=$(docker image inspect pm25-m3:gpu --format '{{.Id}}')
echo $IMAGE_ID
```

- Expect a long build and, for the GPU image, a large image. Build before the timed tests, and note how long the VM has been running.
- The image ID is recorded on every run as `image_digest`. A registry digest exists only after pushing to a registry (for example
  Artifact Registry); the image ID is what this build can record.
- The Linux XGBoost wheel is expected to include CUDA support. The benchmark and `train.py` verify it: each asks XGBoost which device it
  really used and stops if a `cuda` run silently fell back to the CPU, so a CPU run can never be recorded as a GPU run.

## 9. Check first, then run

Define the run command once in the shell. On the CPU VM:

```
RUN="docker run --rm -v $PWD/processed_data:/app/processed_data:ro -v $PWD/mlruns:/app/mlruns -v $PWD/reports:/app/reports -e DATASET_RUN_ID=<DATASET_RUN_ID> -e IMAGE_DIGEST=$IMAGE_ID -e RUN_ENVIRONMENT=gcp-cpu-vm pm25-m3:cpu"
```

On the GPU VM:

```
RUN="docker run --rm --gpus all -v $PWD/processed_data:/app/processed_data:ro -v $PWD/mlruns:/app/mlruns -v $PWD/reports:/app/reports -e DATASET_RUN_ID=<DATASET_RUN_ID> -e IMAGE_DIGEST=$IMAGE_ID -e RUN_ENVIRONMENT=gcp-gpu-vm pm25-m3:gpu"
```

**9a. A few-minute check (its own experiment, not a result), on both VMs:**

```
$RUN src/benchmark.py --config configs/compute_benchmark_check.yaml --machine-type n1-standard-8 --region <REGION> --physical-cores 4
```

On the GPU VM use `--machine-type n1-standard-8-t4`. On the GPU VM it must finish without a SKIPPED line for `cuda`, and the summary must list
`cuda` rows for `xgb_device` and `gru_device`; if a `cuda` test is skipped, the GPU is not visible to the container, so go back to step 6. On
the CPU VM two SKIPPED lines for `cuda` are expected. Delete the check's folder afterwards
(`rm -r reports/m3/m3_benchmark_check`) so it is not copied back as a result.

**9b. The CPU VM tests** (nothing else running on the VM):

```
$RUN src/benchmark.py --config configs/compute_benchmark.yaml --machine-type n1-standard-8 --region <REGION> --physical-cores 4 --hourly-rate <RATE_CPU>
```

This runs all three tests on the CPU (the `cuda` rows are skipped and said so). The tuned XGBoost needs about 3,700 rounds per fit, and the
thread-scaling test fits it 6 times (a discarded warm-up and 5 repeats) at each of 4 thread counts, so this is the longest step: the tuned
8-thread fit took about 100 seconds on the laptop, so expect on the order of an hour or more (an estimate, not measured on a VM).

**9c. The GPU VM tests** (the rates: the GPU machine, and the same machine without the GPU for the CPU rows):

```
$RUN src/benchmark.py --config configs/compute_benchmark.yaml --only xgb_device gru_device --machine-type n1-standard-8-t4 --region <REGION> --physical-cores 4 --hourly-rate <RATE_GPU> --cpu-only-rate <RATE_CPU>
```

**9d. The GRU profile:**

```
$RUN src/profile_gru.py --config configs/profile_gru.yaml --device cpu     # CPU VM
$RUN src/profile_gru.py --config configs/profile_gru.yaml --device cuda    # GPU VM
```

**9e. The reproduction check** (the tuned XGBoost; accepted when the selection-fold MAE is within 0.005 and the 2024 MAE within 0.003 of
3.3393 and 3.1466, `docs/MODEL_CARD.md` section 11):

```
$RUN src/train.py --config configs/reproduce_xgboost_tuned_cpu.yaml                       # both VMs
$RUN src/train.py --config configs/reproduce_xgboost_tuned_gpu.yaml                       # GPU VM only
```

A GPU build differs numerically from a CPU build: report any difference. The CPU model stays the frozen model.

## 10. Record what the report needs

Run these on each VM and copy the output into your notes:

```
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv      # GPU VM only
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
| Time each VM was running, and the approximate cost | VM start and stop times × the rate |

The results are written as plain files: `reports/m3/m3_benchmark/<environment>/` (repeats, summary and `machine.json`), `reports/m3/profiling/<device>/`
and the logged reproduction runs. Export the run tables on each VM, then copy the results back **from the laptop** into the repository's
`reports/` folder, where `plot_results.py` finds them:

```
$RUN src/export_runs.py
```

```
gcloud compute scp --recurse pm25-bench-cpu:~/dh-PM2.5-forecast/reports/m3/m3_benchmark/gcp-cpu-vm ./reports/m3/m3_benchmark/ --zone=<ZONE> --project=<PROJECT_ID>
gcloud compute scp --recurse pm25-bench-gpu:~/dh-PM2.5-forecast/reports/m3/m3_benchmark/gcp-gpu-vm ./reports/m3/m3_benchmark/ --zone=<ZONE> --project=<PROJECT_ID>
gcloud compute scp --recurse pm25-bench-cpu:~/dh-PM2.5-forecast/reports/m3/profiling/cpu ./reports/m3/profiling/cpu_vm --zone=<ZONE> --project=<PROJECT_ID>
gcloud compute scp --recurse pm25-bench-gpu:~/dh-PM2.5-forecast/reports/m3/profiling/cuda ./reports/m3/profiling/cuda --zone=<ZONE> --project=<PROJECT_ID>
gcloud compute scp --recurse pm25-bench-cpu:~/dh-PM2.5-forecast/reports/m3/run_summary.csv ./reports/m3_gcp/cpu_vm_run_summary.csv --zone=<ZONE> --project=<PROJECT_ID>
gcloud compute scp --recurse pm25-bench-gpu:~/dh-PM2.5-forecast/reports/m3/run_summary.csv ./reports/m3_gcp/gpu_vm_run_summary.csv --zone=<ZONE> --project=<PROJECT_ID>
```

(Create `reports/m3_gcp` first.) The VMs' MLflow databases record their container paths, so view the results from these files, not by opening a
VM's database on the laptop. Run the same benchmark on the laptop with `RUN_ENVIRONMENT=laptop`, natively in the project's virtual environment (the pinned versions and Python 3.14.2 are the same as in the image, and the MLflow artefact paths of the laptop runs stay valid on the laptop; a container run would record container paths in the laptop's store). The folder `reports/m3/m3_benchmark/laptop/` is then written in place. Then draw the comparison:

```
python src/plot_results.py --figure benchmark
```

## 11. Stop and delete everything

As soon as the results are on the laptop, from the laptop:

```
gcloud compute instances delete pm25-bench-cpu pm25-bench-gpu --zone=<ZONE> --project=<PROJECT_ID> --delete-disks=all
gcloud compute instances list --project=<PROJECT_ID>
gcloud compute disks list --project=<PROJECT_ID>
```

Both lists must be empty of benchmark resources. A stopped VM still pays for its disk, so delete rather than stop unless a rerun is
certain. Then check Billing → Reports the next day for the actual charge (filter by the labels), and write it in the report beside the estimate.

## Troubleshooting

| Symptom | Likely cause and action |
|---|---|
| `gcloud` reports quota exceeded for `NVIDIA_T4_GPUS` or `GPUS_ALL_REGIONS` | Step 1: the limit is still 0; wait for the increase |
| Creating a VM fails with a resource-availability or "zone does not have enough resources" error | The zone has no free capacity right now; try another zone from step 2, for both VMs |
| `nvidia-smi` not found after step 5 | The installer was interrupted by the restart; reconnect and run `install_driver` again |
| `docker: Error response ... could not select device driver "nvidia"` | The container toolkit is not connected to Docker; repeat the `nvidia-ctk runtime configure` step and restart Docker |
| A run stops with "XGBoost was asked for device='cuda' but ran on 'cpu'" | XGBoost cannot see the GPU in the container; check `docker run --gpus all ... nvidia-smi` and that the image was built with the GPU `TORCH_INDEX` |
| A hash in step 7 does not match | The files are not the dataset the experiments used; stop and check the bucket's `manifests/dataset_versions.jsonl` |
| `permission denied` reading the bucket | The VM's service account lacks `roles/storage.objectViewer` on the bucket (step 3) |
| `reports/m3/m3_benchmark/<environment>` is missing after a run | `RUN_ENVIRONMENT` was not passed to the container (`-e RUN_ENVIRONMENT=...`); the folder is then named after `--machine-type` |

## Sources

- [Install GPU drivers (Compute Engine)](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu)
- [Create an N1 VM that has attached GPUs](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose)
- [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
