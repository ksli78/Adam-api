# Ollama Model Management Guide

Complete guide for managing Ollama models and disk space.

---

## Understanding Ollama Storage

### How Ollama Stores Models

**Docker Volume:**
- Models are stored in a Docker volume (not in the container)
- Volume name: `ollama-models`
- Location: `/var/lib/docker/volumes/ollama-models/`
- Persists even when containers are removed

**Shared Models:**
- When using dual Ollama (ollama-gpu0 and ollama-gpu1), both containers **share** the same volume
- Pull a model once, both containers use it
- Very efficient - no duplicate downloads!

**Model Sizes:**
- Mistral-Small 22B: ~12GB
- Mistral 7B: ~4GB
- Qwen2.5 7B: ~4GB
- Llama 3.2 3B: ~2GB
- Phi-3 Mini: ~2.3GB

---

## List Your Current Models

### From Command Line

```bash
# If using single Ollama container
docker exec ollama ollama list

# If using dual Ollama
docker exec ollama-gpu0 ollama list
# (ollama-gpu1 will show the same - shared storage)
```

**Example output:**
```
NAME                     ID              SIZE    MODIFIED
mistral-small:22b       abc123def456    12 GB   2 days ago
mistral:7b-instruct     def789ghi012    4.1 GB  5 hours ago
```

---

## Remove Individual Models

### Remove Specific Model

```bash
# Remove Mistral-Small 22B (frees ~12GB)
docker exec ollama-gpu0 ollama rm mistral-small:22b

# Remove any model by name
docker exec ollama-gpu0 ollama rm <model-name>
```

**What happens:**
- Model files are deleted from the volume
- Space is freed immediately
- Both ollama-gpu0 and ollama-gpu1 are affected (shared storage)

---

## Automated Cleanup Script

### Interactive Cleanup

Use the provided cleanup script:

```bash
# Make executable
chmod +x cleanup-ollama-models.sh

# Run interactive cleanup
./cleanup-ollama-models.sh
```

**What it does:**
1. Lists all your current models with sizes
2. Shows total storage usage
3. Lets you select which models to remove
4. Option to remove ALL models for fresh start

**Example session:**
```
Current Models
====================
NAME                     ID              SIZE    MODIFIED
mistral-small:22b       abc123          12 GB   2 days ago
mistral:7b-instruct     def456          4.1 GB  1 hour ago

Available models to remove:
  [1] mistral-small:22b (12 GB)
  [2] mistral:7b-instruct (4.1 GB)
  [0] Remove ALL models (fresh start)
  [Q] Quit

Select models to remove (comma-separated, e.g., 1,3 or 0 for all): 1
```

---

## Check Disk Space Usage

### Docker Volume Size

```bash
# List all Docker volumes
docker volume ls

# Inspect Ollama models volume
docker volume inspect ollama-models

# Get volume size (need sudo)
sudo du -sh /var/lib/docker/volumes/ollama-models/
```

### Overall Docker Disk Usage

```bash
# Show Docker disk usage summary
docker system df

# Detailed breakdown
docker system df -v
```

**Example output:**
```
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          10        5         15.2GB    8.1GB (53%)
Containers      8         3         1.2GB     800MB (66%)
Local Volumes   5         3         28GB      12GB (42%)
Build Cache     0         0         0B        0B
```

---

## Clean Up Dual Ollama Deployment

### Recommended Cleanup Before Deployment

**Step 1: List current models**
```bash
docker exec ollama ollama list
```

**Step 2: Remove old large models**
```bash
# Remove Mistral-Small 22B (saves ~12GB)
docker exec ollama ollama rm mistral-small:22b

# Remove any other unused models
docker exec ollama ollama rm <model-name>
```

**Step 3: Deploy dual Ollama**
```bash
./deploy-dual-ollama.sh
```

**Result:**
- Old container removed: `ollama`
- New containers created: `ollama-gpu0`, `ollama-gpu1`
- Old volume reused (keeps any models you didn't remove)
- New model downloaded: `mistral:7b-instruct-v0.3` (~4GB)

---

## Cleanup After Deployment

### Verify Shared Storage

After deploying dual Ollama, both containers use the same models:

```bash
# Check GPU 0 models
docker exec ollama-gpu0 ollama list

# Check GPU 1 models (should be identical)
docker exec ollama-gpu1 ollama list
```

**Both will show:**
```
NAME                     ID              SIZE    MODIFIED
mistral:7b-instruct     abc123          4.1 GB  Just now
```

**If you see duplicate models, something went wrong!**

---

## Complete Docker Cleanup (Advanced)

### Free Maximum Disk Space

**⚠️ WARNING: This removes ALL unused Docker resources!**

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes (except those in use)
docker volume prune

# Remove build cache
docker builder prune -a

# Or do everything at once
docker system prune -a --volumes
```

**This will remove:**
- All stopped containers
- All unused images
- All unused volumes
- All build cache

**Before running, make sure:**
- Your Ollama containers are running
- You've pulled all models you need
- You don't need old containers/images

---

## Model Migration Workflow

### Safe Migration from Single to Dual Ollama

**Option 1: Keep Old Model (No Cleanup)**

```bash
# Deploy dual Ollama
./deploy-dual-ollama.sh

# Both containers will see old Mistral-Small 22B
docker exec ollama-gpu0 ollama list
# Shows: mistral-small:22b (12 GB)

# Pull new Mistral 7B
docker exec ollama-gpu0 ollama pull mistral:7b-instruct-v0.3
docker exec ollama-gpu1 ollama pull mistral:7b-instruct-v0.3

# Now you have BOTH models (uses 16GB total)
docker exec ollama-gpu0 ollama list
# Shows: mistral-small:22b (12 GB)
#        mistral:7b-instruct-v0.3 (4 GB)

# Remove old model when ready
docker exec ollama-gpu0 ollama rm mistral-small:22b
```

**Option 2: Clean Start (Recommended)**

```bash
# Remove old model first (saves space during download)
docker exec ollama ollama rm mistral-small:22b

# Deploy dual Ollama
./deploy-dual-ollama.sh
# Script will download Mistral 7B fresh (~4GB)

# Result: Only 4GB used
```

---

## Troubleshooting

### Model Still Shows After Removal

**Issue:** Removed model still appears in list

**Fix:**
```bash
# Restart Ollama containers
docker restart ollama-gpu0 ollama-gpu1

# Check again
docker exec ollama-gpu0 ollama list
```

---

### Volume Not Freeing Space

**Issue:** Removed models but disk space not freed

**Check:**
```bash
# Check if volume is still in use
docker volume inspect ollama-models

# Check actual disk usage
sudo du -sh /var/lib/docker/volumes/ollama-models/

# If still large, restart Docker
sudo systemctl restart docker
```

---

### Different Models on Each GPU

**Issue:** ollama-gpu0 and ollama-gpu1 show different models

**Cause:** Not using shared volume correctly

**Fix:**
```bash
# Stop containers
docker stop ollama-gpu0 ollama-gpu1

# Check docker-compose.yml volume configuration
grep -A 2 "volumes:" docker-compose.dual-ollama.yml

# Should show:
#   volumes:
#     - ollama-models:/root/.ollama

# Restart
docker compose -f docker-compose.dual-ollama.yml up -d ollama-gpu0 ollama-gpu1
```

---

## Best Practices

### 1. Regular Cleanup

```bash
# Monthly cleanup
./cleanup-ollama-models.sh

# Remove models you're not using
docker exec ollama-gpu0 ollama rm <unused-model>
```

### 2. Monitor Disk Space

```bash
# Check volume size weekly
sudo du -sh /var/lib/docker/volumes/ollama-models/

# Check overall Docker usage
docker system df
```

### 3. Only Keep Models You Use

**Recommended setup:**
- Keep: 1-2 models you actively use (~4-8GB)
- Remove: Everything else

**Example:**
```bash
# Keep only Mistral 7B
docker exec ollama-gpu0 ollama list
docker exec ollama-gpu0 ollama rm <all-other-models>
```

### 4. Use Smaller Models

Instead of keeping multiple large models, use one smaller model:

- Mistral-Small 22B: 12GB (large, slow)
- Mistral 7B: 4GB (medium, fast) ✅ Recommended
- Qwen2.5 7B: 4GB (medium, fast, good for RAG) ✅
- Llama 3.2 3B: 2GB (small, very fast)

---

## Summary

**Model Storage:**
- Stored in Docker volume `ollama-models`
- Shared between ollama-gpu0 and ollama-gpu1
- Persists even when containers are removed

**Cleanup Commands:**
```bash
# List models
docker exec ollama-gpu0 ollama list

# Remove specific model
docker exec ollama-gpu0 ollama rm mistral-small:22b

# Interactive cleanup
./cleanup-ollama-models.sh

# Check disk usage
docker system df
sudo du -sh /var/lib/docker/volumes/ollama-models/
```

**Recommended Before Dual Deployment:**
1. Remove Mistral-Small 22B (saves 12GB)
2. Deploy dual Ollama
3. Use Mistral 7B (only 4GB)
4. Total savings: 8GB!

**Disk Space Savings:**
- Mistral-Small 22B: ~12GB
- Mistral 7B: ~4GB
- **You save 8GB** by switching models!
