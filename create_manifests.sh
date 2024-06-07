DATA_ROOT=/Users/tomalcorn/Documents/University/pg/diss/code/data/
LANG_ID=es
AUDIO_MANIFEST_ROOT=/Users/tomalcorn/Documents/University/pg/diss/code/data/manifests

python -m examples.speech_synthesis.preprocessing.get_common_voice_audio_manifest \
  --data-root ${DATA_ROOT} \
  --lang ${LANG_ID} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT} --convert-to-wav