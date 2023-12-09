from get_palette import extract_colors
from palette_to_image import debug_palette

num_of_colors = 5

count = 1000


def map(element):
    img = element["image"]
    main_colors = extract_colors(img, num_of_colors)
    palette = debug_palette(main_colors, size=(50, 20))

    return {"image": element["image"], "palette": palette, "text": element["caption"]}


from functools import partial
from datasets import Dataset
import datasets

dataset = datasets.load_dataset("laion/dalle-3-dataset", streaming=True, split="train")

dataset = dataset.shuffle(seed=42, buffer_size=1_000)

dataset = dataset.map(
    map, remove_columns=["link", "caption", "message_id", "timestamp"]
)


def gen2(e):
    yield from e


ds = Dataset.from_generator(partial(gen2, dataset.take(count)))

ds.push_to_hub("GabrielVidal/dalle-3-palette")
