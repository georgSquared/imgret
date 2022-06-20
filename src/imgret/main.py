import os
import pathlib
import pprint

import cv2 as cv
import typer
from sklearn.metrics import pairwise_distances

from imgret.database import engine, Base
from imgret.models import Image
from imgret.utils import get_image_features, coin_toss, Distances

app = typer.Typer()


@app.callback(invoke_without_command=True)
def callback():
    """
    Image Retrieval Application
    """
    Base.metadata.create_all(bind=engine)


@app.command()
def init(
    dataset_path: str = typer.Argument(
        ...,
        help="Provide a path to the folder containing the images to be loaded into the db",
    ),
    sample: float = typer.Option(
        1.0, "--sample", "-s", help="Only insert a sample of the images"
    ),
):
    """
    Initialise the application with the dataset provided by the user
    :param sample:
    :param dataset_path:
    :return:
    """

    for image_file_name in os.listdir(dataset_path):
        if not coin_toss(sample):
            continue

        image_file_path = os.path.join(dataset_path, image_file_name)
        image_obj = cv.imread(image_file_path, 0)

        typer.echo(f"Adding {image_file_name} to the database")
        try:
            Image.add_or_update(
                features=get_image_features(image_obj),
                file_path=image_file_path,
                file_name=image_file_name,
            )
        except Exception:
            continue


@app.command()
def add(img_path: str = typer.Argument(..., help="The image path")):
    """
    Add a new image to the database.
    """
    path = pathlib.Path(img_path)

    if not path.is_file():
        typer.echo("Invalid file path")
        raise typer.Exit()

    filename = pathlib.Path(img_path).name
    input_img = cv.imread(img_path, 0)

    if input_img is None:
        typer.echo("Invalid image")
        raise typer.Exit()

    Image.add_or_update(
        features=get_image_features(input_img), file_path=img_path, file_name=filename
    )


@app.command()
def search(
    img_path: str = typer.Argument(..., help="The image path"),
    k: int = typer.Option(
        5, "--k", "-k", help="Provide the desired amount of images retrieved"
    ),
    dist: Distances = typer.Option(
        Distances.sqeuclidean,
        "--distance",
        "-dm",
        help="Select the desired distance measure",
    ),
):
    """
    Search for relevant images based on given measure
    """
    input_img = cv.imread(img_path, 0)

    results = Image.get_relevant(input_img, k=k, distance=dist.value)

    pprint.pprint(results)


@app.command()
def test(img_path: str):
    input_img = cv.imread(img_path, 0)
    features = get_image_features(input_img)

    db_image = Image.manager.get(1)
    db_features = db_image.features

    if db_features.shape[0] > features.shape[0]:
        features.resize(db_features.shape)
    else:
        db_features.resize(features.shape)

    dist = pairwise_distances(
        features.reshape(1, -1), db_features.reshape(1, -1), metric="cosine"
    )
    typer.echo(dist)
