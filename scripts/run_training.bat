@echo off
cd /d F:\editorbot
python -m pointer_network.trainers.pointer_trainer --cache-dir cache --pointer-dir training_data/pointer_sequences --save-dir models/pointer_network --epochs 100 --no-compile --batch-size 8 --resume models/pointer_network/best.pt >> F:\editorbot\logs\training_output.log 2>&1
