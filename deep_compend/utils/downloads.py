"""File downloading module."""


def download_arxiv_paper(
    arxiv_id: str, save_path: str, chunk_size: int = 8192
) -> None:
    """Downloads a PDF directly from ArXiv given its ID.

    Args:
        arxiv_id (str): Arxiv paper ID.
        save_path (str): Path where to save a downloaded paper.
        chunk_size (int, optional): Number of bytes to be read when downloading. Defaults to 8192.
    """
    import requests

    # Setting URL from where to download a paper
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        response = requests.get(url, stream=True, timeout=10)
        if (
            response.status_code == 200
            and "application/pdf" in response.headers.get("Content-Type", "")
        ):
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            print(
                f"Paper with ID '{arxiv_id}' is downloaded and saved in '{save_path}'"
            )
        else:
            print(
                f"Paper with ID '{arxiv_id}' not found. HTTP Status: {response.status_code}"
            )
    except requests.RequestException as e:
        print(f"Error downloading paper: {e}")


def ensure_nltk_resource(resource_id: str) -> None:
    """Looks for NLTK resource and downloads it if not present.

    Args:
        resource_id (str): Name of NLTK resource.
    """
    import nltk

    try:
        nltk.data.find(resource_id)
    except LookupError:
        nltk.download(resource_id.split("/")[1], quiet=True)
