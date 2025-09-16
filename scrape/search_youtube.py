"""
Performs a search on YoutTube and retrieves
"""

from googleapiclient.discovery import build
from googleapiclient.discovery import Resource
from scrape.search_terms import pokemon_search_keywords as search_keywords
import os
import argparse
import json

api_key = os.environ["YOUTUBE_KEY"]
youtube: Resource = build("youtube", "v3", developerKey=api_key)


def search(keyword: str, num_pages: int):
    search_results = []
    next_page_token = None

    for page_number in range(num_pages):
        # API will pull at most 50 results per page
        request = youtube.search().list(
            part="snippet",
            q=keyword,
            type="video",
            maxResults=50,
            pageToken=next_page_token,
        )
        response = request.execute()

        for item in response["items"]:
            search_results.append(
                {
                    "search_result": item,
                    "metadata": {"keyword": keyword, "page_number": page_number},
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return search_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default=1, type=int)
    parser.add_argument("--terms", default=-1, type=int)
    parser.add_argument("--output_dir", default="./cache/search_results.json", type=str)

    args = parser.parse_args()

    if args.terms > 0:
        search_keywords = search_keywords[: args.terms]

    search_results = []

    for keyword in search_keywords:
        search_results.extend(search(keyword, args.pages))

    output_dir_path = os.path.dirname(args.output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(args.output_dir, "w") as f:
        json.dump(search_results, f, indent=2)
