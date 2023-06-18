# Swin Transformer Classification Trainner

- Swin transformer classification trainner.

- MyDataset class just for voc dataset with has xml annotation, for there's may some broken image file in my own dataset, I set exception catches with image randomly get.

- Use huggingface transformer and model file, so it can change to any model else easily.


SWTClassification.py: based on huggingface accelerate

SWTClassification_dist.py: based on pytorch distributed