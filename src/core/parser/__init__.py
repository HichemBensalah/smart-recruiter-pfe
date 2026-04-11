from .docling_parser import DoclingParser


def parse_cv(file_path: str):
    parser = DoclingParser()
    return parser.parse(file_path)
