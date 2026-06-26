from ddgs import DDGS

def search(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)

        for i, r in enumerate(results, 1):
            print(f"\nResult {i}")
            print("Title:", r["title"])
            print("Link:", r["href"])
            print("Snippet:", r["body"])


if __name__ == "__main__":
    while True:
        q = input("\nEnter search query (or 'exit'): ")
        if q.lower() == "exit":
            break
        search(q)