#!/bin/bash

# Sanal ortamı aktif et
source .venv/bin/activate

# Logs klasörünü oluştur
mkdir -p logs

# Embedding klasörlerinin listesi
EMBEDDINGS=("convnext" "dinov2" "dinov3" "openclip" "siglip")

echo "Batch training starting..."

for embedding in "${EMBEDDINGS[@]}"; do
    echo "[$(date)] Starting training for $embedding..."
    
    # Python scriptini çalıştır
    # Logları logs/ klasörüne kaydet
    python src/train_mlp.py --embedding_dir "embeddings/$embedding" > "logs/mlp_${embedding}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] Successfully finished $embedding"
    else
        echo "[$(date)] Failed training $embedding"
    fi
done

echo "[$(date)] All trainings completed."

