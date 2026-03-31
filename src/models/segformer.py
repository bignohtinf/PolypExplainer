from transformers import SegformerForSemanticSegmentation, SegformerConfig

def get_segformer_model(model_name, num_classes=1):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={0: "background", 1: "polyp"},
        label2id={"background": 0, "polyp": 1},
        ignore_mismatched_sizes=True,
    )
    return model
