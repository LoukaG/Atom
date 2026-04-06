import json
import os


def load_all_bot_ids(paths):
    bot_ids = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    bot_ids.add(line)
    return bot_ids


def load_bot_ids(path: str) -> set:
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def load_dataset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_posts_by_user(posts):
    grouped = {}
    for post in posts:
        uid = post["author_id"]
        grouped.setdefault(uid, []).append(post)
    return grouped


def load_all_datasets(paths, bot_ids):
    all_users = []

    for idx, path in enumerate(paths, 1):
        print(f"Loading dataset {idx}/{len(paths)}...")
        data = load_dataset(path)
        users = data["users"]
        posts_by_user = group_posts_by_user(data["posts"])

        for user in users:
            uid = user["id"]
            all_users.append(
                {
                    "user": user,
                    "posts": posts_by_user.get(uid, []),
                    "label": int(uid in bot_ids),
                    "source": os.path.basename(path),
                }
            )

    return all_users
