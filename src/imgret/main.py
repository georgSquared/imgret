import typer


app = typer.Typer()


@app.callback()
def callback():
    """
    Image Retrieval Application
    """


@app.command()
def add():
    """
    Add a new image to the database
    """
    typer.echo("Added image with vectors")


@app.command()
def search():
    """
    Search for relevant images based on given measure
    """

    typer.echo("Searching for relevant images")
