N_CLUSTERS=100
TYPE=hubert
CKPT_PATH=/Users/tomalcorn/Documents/University/pg/diss/code/models/hubert_base_ls960.pt
LAYER=6
MANIFEST=<tab_separated_manifest_of_audio_files_for_training_kmeans>
KM_MODEL_PATH=models/kmeans.ptv

PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
    --num_clusters $N_CLUSTERS \
    --feature_type $TYPE \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_kmeans_model_path $KM_MODEL_PATH