import json 
import glob 
import argparse
import re 
import datetime 


DOI_PATTERN = r"\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>])\S)+)\b"


def getAllMetadata():
    """
    Loads all of the metadata files in /qml/demonstrations and returns them as a dictionary, where the keys are the file names minus the endings.
    """

    metadatas = {}
    filePaths = glob.glob("demonstrations/*.metadata.json")

    for filePath in filePaths:
        i = filePath.find(".metadata")
        fileName = filePath[:i]

        with open(filePath, "r", encoding="utf-8") as fo:
            metadata = json.load(fo)

            metadatas[fileName] = metadata 

    return metadatas 


def checkMetadata():
    """
    Goes through all of the metadata files and runs checks against them. Feel free to change the checks that are done.
    """

    metadatas = getAllMetadata()

    for name, metadata in metadatas.items():
        if not metadata["seoDescription"].endswith("."):
            print("{0} is missing a full stop at the end of its description.".format(name))
        if len(metadata["categories"]) == 0:
            print("{0} is not in any category.".format(name))

        for doi in metadata["basedOnPapers"]:
            if doi != "" and not re.match(DOI_PATTERN, doi):
                print("{0} has an incorrectly-formatted DOI.".format(name))

        for reference in metadata["references"]:
            doi = reference.get("doi", "")
            
            if doi != "" and not re.match(DOI_PATTERN, doi):
                print("{0} has an incorrectly-formatted DOI.".format(name))


def retitleCategory(fromTitle, toTitle):
    """
    Changes the title of a category.
    """

    fps = glob.glob("./demonstrations/*.metadata.json")

    for fp in fps:
        with open(fp, "r", encoding="utf-8") as fo:
            metadata = json.load(fo)

        metadata["categories"] = [toTitle if c.strip() == fromTitle else c.strip() for c in metadata["categories"]]

        with open(fp, "w", encoding="utf-8") as fo:
            json.dump(metadata, fo, indent=4, ensure_ascii=False)


def getAllCategoriesUsed():
    """
    Gets all of the categories used in metadata files, prints them, and returns them.
    """

    metadatas = getAllMetadata()
    categories = {}

    for metadata in metadatas:
        for category in metadata["categories"]:
            if category.strip() != "":
                categories[category] = category 

    print([k for k, v in categories.items()])

    return categories


def getMostRecentDemos(n):
    """
    Gets the n most recent demos, prints their names and publication dates, and returns them.
    """

    metadatas = getAllMetadata()

    mostRecent = [v for k, v in metadatas.items()]
    mostRecent = sorted(mostRecent, key=lambda m: datetime.datetime.strptime(m["dateOfPublication"], "%Y-%m-%dT%H:%M:%S"), reverse=True)

    n = len(mostRecent) if n > len(mostRecent) else n 

    for metadata in mostRecent[:n]:
        print("{0}, {1}".format(metadata["title"], metadata["dateOfPublication"]))

    return mostRecent[:n]


def getNumberOfDemosPerYear():
    """
    Counts how many demonstrations were published each year, prints it, and returns it.
    """
    metadatas = getAllMetadata()
    perYear = []

    for year in [2018, 2019, 2020, 2021, 2022, 2023]:
        n = len([d for k, d in metadatas.items() if d["dateOfPublication"].startswith(str(year))])

        perYear.append({"Year": year, "Count": n})

    print("The number of demonstrations published per year:")

    for year in perYear:
        print("{0}: {1}".format(year["Year"], year["Count"]))

    return perYear 



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--action")
    parser.add_argument("--title-1")
    parser.add_argument("--title-2")

    arguments = parser.parse_args()

    if arguments.action == "count":
        """
        Counts the number of metadata files in the system and prints it.
        """
        metadatas = getAllMetadata()
        n = len(metadatas)

        print("There are {0} metadata files in total.".format(n))
    
    if arguments.action == "count_per_year":
        getNumberOfDemosPerYear()

    if arguments.action == "check":
        checkMetadata()

    if arguments.action == "retitle_category":
        title1 = arguments.title_1.strip()
        title2 = arguments.title_2.strip()

        retitleCategory(title1, title2)

    if arguments.action == "get_all_categories_used":
        getAllCategoriesUsed()

    if arguments.action == "get_most_recent_demos":
        getMostRecentDemos(5)








