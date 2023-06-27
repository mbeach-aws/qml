import json
import io
import argparse
import bibtexparser 
from datetime import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path")
    parser.add_argument("output_file_path")

    arguments = parser.parse_args() 

    inputFilePath = arguments.input_file_path
    outputFilePath = arguments.output_file_path 

    metadata = {}
    metadata["title"] = ""
    metadata["authors"] = []
    metadata["dateOfPublication"] = datetime.strftime(datetime.now(), "%Y-%m-%dT00:00:00+00:00")
    metadata["dateOfLastModification"] = datetime.strftime(datetime.now(), "%Y-%m-%dT00:00:00+00:00")
    metadata["categories"] = []
    metadata["tags"] = []
    metadata["previewImages"] = []
    metadata["seoDescription"] = ""
    metadata["doi"] = ""
    metadata["canonicalURL"] = ""
    metadata["references"] = []
    metadata["basedOnPapers"] = []
    metadata["referencedByPapers"] = []
    metadata["relatedContent"] = []

    with open(inputFilePath, "r", encoding="utf-8") as ifo:
        bibliography = bibtexparser.load(ifo)

        for entry in bibliography.entries:
            reference = {}
            reference["id"] = entry["ID"]
            reference["title"] = entry["title"]
            reference["authors"] = entry["author"]
            reference["year"] = entry["year"]

            if "doi" in entry:
                reference["doi"] = entry["doi"]

            if "url" in entry:
                reference["url"] = entry["url"]

            if "journal" in entry:
                reference["journal"] = entry["journal"]

            if "volume" in entry:
                reference["volume"] = entry["volume"]

            if "number" in entry:
                reference["number"] = entry["number"]

            metadata["references"].append(reference)

    with open(outputFilePath, "w", encoding="utf-8") as ofo:
        json.dump(metadata, ofo, indent=4)