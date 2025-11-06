# Getting Ollama Container to See GPUs on RHEL9 (Portainer)

## Current Situation
- ✅ Host: RHEL9 with `nvidia-smi` working (2 GPUs, 32GB VRAM)
- ✅ Portainer managing Docker containers
- ❌ Ollama container: Cannot see GPUs (no nvidia-smi inside container)

## Goal
Get Ollama container to see and use both GPUs for llama3.1:70b model.

---

## Part 1: Install NVIDIA Container Toolkit on RHEL9 Host

### Step 1: SSH into the Host Server

```bash
ssh user@adam.amentumspacemissions.com
```

### Step 2: Verify GPUs Work on Host

```bash
nvidia-smi
```

**Expected:** Should show 2 GPUs with ~16GB VRAM each.

### Step 3: Install NVIDIA Container Toolkit

```bash
# Add NVIDIA Container Toolkit repository for RHEL9
sudo dnf config-manager --add-repo https://nvidia.github.io/libnvidia-container/rhel9.0/libnvidia-container.repo

# Install the toolkit
sudo dnf install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker

# Verify installation
nvidia-ctk --version
```

**Expected output:** Version number like `1.14.0` or similar.

---

## Part 2: Recreate Ollama Container with GPU Access in Portainer

### Step 4: Open Portainer Web UI

Open your browser and go to Portainer:
```
https://adam.amentumspacemissions.com:9443
```
(or whatever port your Portainer is on)

### Step 5: Stop and Note Current Ollama Container Settings

1. Go to **Containers** in the left menu
2. Find your **ollama** container
3. Click on the container name
4. **Note down these settings** (you'll need them):
   - Port mappings (probably `11434:11434`)
   - Volume mappings (probably `ollama:/root/.ollama`)
   - Environment variables (if any)
5. Click **Stop** button (top right)
6. Once stopped, click **Remove** button

**Don't worry** - Your models are stored in the `ollama` volume and won't be deleted!

### Step 6: Create New Container with GPU Access

1. In Portainer, go to **Containers**
2. Click **Add container** button (top right)

### Step 7: Fill in Container Configuration

#### Basic Settings:
- **Name:** `ollama`
- **Image:** `ollama/ollama:latest`

#### Network ports:
Click **+ map additional port**
- **host:** `11434`
- **container:** `11434`

#### Advanced container settings:

Click **Show advanced options** at the bottom

#### Volumes tab:
Click **+ map additional volume**
- **Volume:** Select `ollama` from dropdown (or create new if missing)
- **Container:** `/root/.ollama`
- **Bind:** Leave as Volume

#### Env tab (Environment variables):
Click **+ add environment variable** twice and add:
- **name:** `OLLAMA_HOST` | **value:** `0.0.0.0:11434`
- **name:** `OLLAMA_ORIGINS` | **value:** `*`

#### Runtime & Resources tab:
This is the **CRITICAL PART** for GPU access:

Scroll down to **Runtime & Resources** section

**GPU Configuration:**
- Enable **GPU access**
- Select **All GPUs** (or manually select both GPUs)

**If you don't see GPU options:**
You need to enable it in Portainer settings:
1. Go to **Settings** (left menu)
2. Click **Edge Compute**
3. Enable **Enable GPU management**
4. Save settings
5. Go back and recreate the container

**Alternative - Use JSON Configuration:**

If GPU options aren't visible, click the **</> Advanced** mode toggle at the top and paste:

```json
{
  "HostConfig": {
    "DeviceRequests": [
      {
        "Driver": "nvidia",
        "Count": -1,
        "Capabilities": [["gpu"]]
      }
    ]
  }
}
```

#### Restart policy:
- Select **Unless stopped**

### Step 8: Deploy Container

Click **Deploy the container** button at the bottom.

Wait 10-20 seconds for it to start.

---

## Part 3: Verify GPU Access

### Step 9: Check Container Logs

In Portainer:
1. Go to **Containers**
2. Click on **ollama** container
3. Click **Logs** button
4. Look for any errors

**Good signs:**
- No "CUDA not available" errors
- No "GPU not found" errors

### Step 10: Open Container Console

In Portainer:
1. Still in the ollama container page
2. Click **Console** button (>_ icon)
3. Select **Command:** `/bin/bash`
4. Click **Connect**

You now have a bash shell inside the container.

### Step 11: Test nvidia-smi Inside Container

In the container console, type:

```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.XX.XX    Driver Version: 535.XX.XX    CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  GPU 0           On       | 00000000:01:00.0 Off |                    0 |
| 30%   40C    P0     50W / 250W |      0MiB / 16384MiB |      0%      Default |
|   1  GPU 1           On       | 00000000:02:00.0 Off |                    0 |
| 30%   40C    P0     50W / 250W |      0MiB / 16384MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

**If you see this:** ✅ **SUCCESS! GPUs are now accessible!**

**If you see "command not found":** ❌ Still no GPU access - go to troubleshooting section below.

---

## Part 4: Test Model Loading with GPU

### Step 12: Load llama3.1:70b Model

Still in the container console:

```bash
# List available models
ollama list

# Try to run the 70b model
ollama run llama3.1:70b "What is 2+2?"
```

### Step 13: Monitor GPU Usage on Host

Open **another SSH session** to the host:

```bash
ssh user@adam.amentumspacemissions.com

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

**Expected behavior:**
- After a few seconds, you should see VRAM usage climb
- GPU 0: ~14GB usage
- GPU 1: ~14GB usage
- Total: ~28GB for the model
- GPU Utilization: 80-100% during inference

### Step 14: Test from Application

From your **application server**, test the API:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "How many hours can I work in a single week?",
  "use_llm_selection": true,
  "max_documents": 10
}'
```

Watch the `nvidia-smi` output on the host - you should see GPU activity!

---

## Troubleshooting

### Issue 1: GPU Options Not Visible in Portainer

**Solution:**

1. Go to Portainer **Settings**
2. Enable GPU management
3. Restart Portainer:
   ```bash
   sudo systemctl restart portainer
   ```

### Issue 2: Still No nvidia-smi in Container After Recreating

**Check Docker daemon configuration:**

```bash
# On host, check Docker daemon config
cat /etc/docker/daemon.json

# Should contain:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# If missing, create it:
sudo tee /etc/docker/daemon.json <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

# Restart Docker
sudo systemctl restart docker

# Restart Portainer (if it's in a container)
docker restart portainer
```

Then recreate the Ollama container (Steps 5-8).

### Issue 3: nvidia-container-toolkit Not Found

```bash
# On RHEL9, ensure you have the correct repository
sudo dnf config-manager --add-repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo

# Update and install
sudo dnf clean all
sudo dnf install -y nvidia-container-toolkit

# Configure
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Issue 4: Model Doesn't Fit in Memory

After GPU access is working, if the model still doesn't fit:

**Option A: Use smaller quantization**
```bash
# In container console
ollama pull llama3.1:70b:q3_K_S  # ~26GB instead of ~40GB
```

**Option B: Use smaller model**
```bash
# In container console
ollama pull llama3.1:13b  # Only ~13GB

# Update application
export LLM_MODEL="llama3.1:13b"
```

---

## Verification Checklist

Before testing with your application, verify:

- [ ] `nvidia-smi` works **on host** ✓
- [ ] NVIDIA Container Toolkit installed on host ✓
- [ ] Docker daemon configured for NVIDIA runtime ✓
- [ ] Ollama container recreated with GPU access ✓
- [ ] `nvidia-smi` works **inside container** ✓
- [ ] `ollama list` shows llama3.1:70b ✓
- [ ] `ollama run llama3.1:70b "test"` works ✓
- [ ] GPU memory usage appears in `nvidia-smi` during inference ✓

---

## Alternative: Docker CLI Method (If Portainer Issues)

If you have trouble with Portainer, you can create the container via CLI:

```bash
# On host, stop/remove existing container
docker stop ollama
docker rm ollama

# Create new container with GPU access
docker run -d \
  --gpus all \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -e OLLAMA_ORIGINS=* \
  --restart unless-stopped \
  ollama/ollama:latest

# Test GPU access
docker exec ollama nvidia-smi
```

Then you can manage it through Portainer afterwards.

---

## Summary of What We're Doing

1. **Host setup:** Install NVIDIA Container Toolkit so Docker can use GPUs
2. **Docker config:** Configure Docker daemon to recognize NVIDIA runtime
3. **Container setup:** Add `--gpus all` flag when creating Ollama container
4. **Verification:** Check `nvidia-smi` works inside container
5. **Test:** Run model and watch GPU usage

The key is the **`--gpus all`** flag (or DeviceRequests in JSON) when creating the container!

---

Let me know at which step you get stuck or what error you see!
