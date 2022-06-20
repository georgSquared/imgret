import os
import pathlib
import pprint

import cv2 as cv
import typer
from sklearn.metrics import pairwise_distances

from imgret.database import engine, Base
from imgret.models import Image, GroundTruth
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
def build_ground_truth(
    ground_truth_path: str = typer.Argument(
        ...,
        help="Provide a path to the folder containing the ground truth files",
    ),
):
    """
    Build a ground truth table to calculate retrieval performance
    This operation is specific to the oxford building dataset
    :return:
    """
    for query_file_name in os.listdir(ground_truth_path):
        if "query" not in query_file_name:
            continue

        fline = (
            open(os.path.join(ground_truth_path, query_file_name)).readline().rstrip()
        )
        query_file_root = fline.split(" ")[0].replace("oxc1_", "")

        good_file_name = query_file_name.replace("query", "good")
        ok_file_name = query_file_name.replace("query", "ok")

        with open(os.path.join(ground_truth_path, good_file_name)) as file:
            lines = file.readlines()
            good_results_list = [line.rstrip() for line in lines]

        with open(os.path.join(ground_truth_path, ok_file_name)) as file:
            lines = file.readlines()
            ok_results_list = [line.rstrip() for line in lines]

        ground_truth_list = good_results_list + ok_results_list

        image = Image.manager.filter(
            Image.file_name == query_file_root + ".jpg"
        ).first()
        if not image:
            continue

        typer.echo(f"Adding ground truth for {image.file_name}")
        for truth_file in ground_truth_list:
            GroundTruth.manager.create(image_id=image.id, file_name=truth_file + ".jpg")


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
    include_metrics: bool = typer.Option(
        True, help="Should a precision metric be calculated?"
    ),
):
    """
    Search for relevant images based on given measure
    """
    input_img = cv.imread(img_path, 0)

    results = Image.get_relevant(input_img, k=k, distance=dist.value)

    typer.secho("****", fg=typer.colors.BLUE)
    typer.secho(f"Top {k} most relevant images:", fg=typer.colors.BLUE)
    typer.secho("****", fg=typer.colors.BLUE)
    for result in results:
        typer.secho(f"Id: {result.get('id')}", fg=typer.colors.GREEN)
        typer.secho(f"File Name: {result.get('file_name')}", fg=typer.colors.GREEN)
        typer.secho(f"File Path: {result.get('file_path')}", fg=typer.colors.GREEN)
        typer.echo("====")

    if include_metrics:
        reference_image = Image.manager.get(results[0]["id"])

        result_set = set()
        for result in results:
            result_set.add(result["file_name"])

        truth_set = set()
        for truth in reference_image.ground_truth:
            truth_set.add(truth.file_name)

        tp_count = len(set(result_set).intersection(truth_set))
        p_count = len(result_set)

        precision = 100 * float(tp_count) / float(p_count)
        typer.secho(f"Precision of results: {precision}", fg=typer.colors.RED)


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
