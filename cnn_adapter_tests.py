import dtlpy as dl
from cnn_adapter import ModelAdapter


def train_test(model: dl.Model):
    model_adapter = ModelAdapter(model_entity=model)
    model_adapter.train_model(model=model)


def predict_test(model: dl.Model):
    model_adapter = ModelAdapter(model_entity=model)
    item = dl.items.get(item_id='645cc2de66671c2da8908f3a')
    result = model_adapter.predict_items(items=[item])

    item_list = result[0]
    item = item_list[0]
    annotations = item.annotations.list()
    label = annotations[0].label
    print(f"Predicted label: {label}")


def main():
    model = dl.models.get(model_id='645f9b256e76507c009c8f95')
    train_test(model=model)
    # predict_test(model=model)


if __name__ == "__main__":
    main()
