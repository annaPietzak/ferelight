import open_clip

model, _, _ = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k')
model.eval()
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')
