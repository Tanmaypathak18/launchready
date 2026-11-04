"""Check docs/index.html and the public tree: links resolve, and banned text stays out.

GitHub Pages serves only docs/, so images must live there and evidence must be absolute links.
The tree check uses generic regexes and hashed words, so this file never names what it bans.

    uv run python scripts/check_site.py            # offline: repo files, anchors, banned text
    uv run python scripts/check_site.py --online   # also fetch every external link
"""

import argparse
import hashlib
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "index.html"
REPO_URL = "https://github.com/Tanmaypathak18/launchready"
# Copies in docs/ of files that must stay identical to the committed original.
ASSETS = {"docs/assets/memory_validation.png": "results/memory_validation.png"}
BANNED = ("native",)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I)
ALLOWED_EMAIL_DOMAINS = ("example.com", "anthropic.com")
# Character classes keep these patterns from matching their own source.
LOCAL_DETAILS = {
    "home directory path": re.compile(r"/[Uu]sers/[^<\s]|/hom[e]/[^<\s]|[A-Z]:\\[Uu]sers"),
    "laptop hostname": re.compile(r"[\w-]+-[Mm]acBook-\w+|@[\w-]+\.loca[l]\b"),
}
# Files outside these prefixes are the project's own writing; these hold raw or third-party text.
RAW_DATA = ("results/", "tests/fixtures/", "uv.lock")
# sha256 of lowercase words, or pairs of adjacent words, that must stay out of the tree.
PRIVATE_WORD_HASHES = {
    "087ac9dbc22f70aaf1db90d1b11c538d64adc0e2fa784204f3a4e852f573013e",
    "31eb78c4e3e8b3d5d50cfb24408d4b7d01aa5f76f4a7409394856400dee4d305",
    "4bb2f659b74005b5ae2f281f0340dc29a902c32d32fe7094a63010dddef25be1",
}
# The only URLs allowed to contain a private word, because the live tool and the article sit at
# them. Only these exact strings are exempt: each must stand alone, so longer URLs and the bare
# word still fail.
ALLOWED_URLS = (
    "https://huggingface.co/spaces/tanmaypathak4/launchready",
    "https://huggingface.co/blog/tanmaypathak4/launchready",
)
_ALLOWED = "|".join(re.escape(url) for url in ALLOWED_URLS)
ALLOWED_URL_MATCH = re.compile(rf"""(?<![^\s(\["'>])(?:{_ALLOWED})(?![^\s)\]"'<,])""")
# The same, for job-search wording, checked only in the project's own writing.
JOB_WORD_HASHES = {
    "02540942346fdfbf21a94f8b823af2ae315f20cb71721c2cc6e205097d560840",
    "2c472cb7f4d7e83f751cf5f17952703f690a0286cc18c8532a471a7b19b3b199",
    "320dc1c2c337e7ef3415e7d56d46b035bf6e0f6639c9691f95590b9cf9e70bc0",
    "44574c4ba2ea74ad4bf1e184133cdbf4e7390a3690beff6a7364511a70ec208e",
    "4a8dba8cb8475d0491e76d6a7c06ac29cbf6c3c6725e8286f3f44be876a6b096",
    "5b8aee8d784120382314f99b140baa7146360aa8806082b631460b548f0401e6",
    "84308bea454057aa509a12fbd5212988973d7cd513bc555a24a06dd2cc72e39e",
    "92d46204d8e9aeb3b37873c794348b3132ab9c9f33a89f2063c46180914c0104",
    "a0b89699b2e6fd46c19bb33a69045d131fa8b46f2b2aa108d97e4ef614b03877",
    "a6ba8d504c4b9320a03ba8cec1ae743153e8fec9ae7077d3b8c9717c5cbc29f5",
    "a83a31320d921b888a48fa5edd0b4b5a29984de6e96bf7b8ac7d29ba06caf616",
    "aafab8f7c6fd0bba239e6fcd7f98260e40aba30b1d11e4da557c5f827b7aabd9",
    "ad3e69e9aa860657cc6476770fe253d08198746b9fcf9dc3186b47eb85c30335",
    "b8fac54e118f435e8b6190abfe473ae4488ab76a7d4c41ce0b512d51ebea7a66",
    "e18ccb8415932de12df9ea563fb47065abe53701ba93b1ce069e1004e8378197",
    "e83c0a7332a6ccfd6e4abd512380cf5d22a8906443dda87e8c53c977ef6c09cb",
    "ec8b778e04b2ff16f644d1d58b86e2f25ce380bcbe1c6eb83e6bffb89a941fba",
    "fb9d78cb1e099ccf5c7b06dca767a1f6b196d719961feb11e0d341049d609164",
}


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []
        self.images: list[str] = []
        self.ids: set[str] = set()
        self.external_code: list[str] = []
        self.text: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        elif tag == "img":
            self.images.append(attrs.get("src", ""))
        elif tag == "script" or (tag == "link" and attrs.get("rel") == "stylesheet"):
            self.external_code.append(tag)
        if tag in ("style", "script"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.text.append(data)


def parse_page(path: Path = PAGE) -> Page:
    page = Page()
    page.feed(path.read_text())
    return page


def heading_anchors(markdown: str) -> set[str]:
    """Anchors GitHub generates for a Markdown file's headings."""
    anchors = set()
    for line in markdown.splitlines():
        if match := re.match(r"#{1,6}\s+(.*)", line):
            title = re.sub(r"[^\w\- ]", "", match[1].strip().lower())
            anchors.add(title.replace(" ", "-"))
    return anchors


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return set(out.splitlines())


def check_repo_link(url: str, tracked: set[str]) -> str | None:
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path.removeprefix(urllib.parse.urlsplit(REPO_URL).path).lstrip("/")
    if not path:
        return None
    kind, _, rest = path.partition("/")
    branch, _, repo_path = rest.partition("/")
    if kind not in ("blob", "tree") or branch != "main":
        return f"not a blob/main or tree/main link: {url}"
    repo_path = urllib.parse.unquote(repo_path)
    if kind == "tree":
        if not any(name.startswith(repo_path + "/") for name in tracked):
            return f"no tracked directory: {repo_path}"
        return None
    if repo_path not in tracked:
        return f"file not tracked: {repo_path}"
    anchor = parsed.fragment
    if not anchor:
        return None
    text = (ROOT / repo_path).read_text()
    if line := re.fullmatch(r"L(\d+)", anchor):
        if parsed.query != "plain=1":
            return f"line anchor needs ?plain=1 on a rendered file: {url}"
        if int(line[1]) > len(text.splitlines()):
            return f"line past end of file: {url}"
    elif anchor not in heading_anchors(text):
        return f"no such heading: {repo_path}#{anchor}"
    return None


def check_site(path: Path = PAGE) -> tuple[list[str], list[str]]:
    """Return (problems, external URLs) without touching the network."""
    page = parse_page(path)
    tracked = tracked_files()
    problems: list[str] = []
    external: list[str] = []

    for link in page.links:
        if link.startswith("#"):
            if link[1:] not in page.ids:
                problems.append(f"no element with id: {link}")
        elif link == REPO_URL or link.startswith(REPO_URL + "/"):
            external.append(link)
            if problem := check_repo_link(link, tracked):
                problems.append(problem)
        elif link.startswith(("http://", "https://")):
            external.append(link)
        elif not (path.parent / link).is_file():
            problems.append(f"missing local file: {link}")

    for src in page.images:
        if "://" in src or src.startswith((".", "/")) or not (path.parent / src).is_file():
            problems.append(f"image must be a file inside docs/: {src}")

    for copy, original in ASSETS.items():
        if (ROOT / copy).read_bytes() != (ROOT / original).read_bytes():
            problems.append(f"{copy} differs from {original}")

    if page.external_code:
        problems.append(f"page must have no scripts or stylesheets: {page.external_code}")
    source = path.read_text().lower()
    problems += [f"banned text: {word}" for word in BANNED if word in source]
    problems += [f"page: {category}" for category in text_problems(source, authored=True)]
    return problems, sorted(set(external))


def text_problems(text: str, authored: bool) -> set[str]:
    """Categories of banned content in a text. Never echoes the match, which may be private."""
    found = {name for name, pattern in LOCAL_DETAILS.items() if pattern.search(text)}
    for address in EMAIL.findall(text):
        if not address.lower().endswith(ALLOWED_EMAIL_DOMAINS):
            found.add("email address")
    words = re.findall(r"[a-z0-9]+", ALLOWED_URL_MATCH.sub(" ", text).lower())
    terms = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:], strict=False)}
    hashes = {hashlib.sha256(term.encode()).hexdigest() for term in terms}
    if hashes & PRIVATE_WORD_HASHES:
        found.add("private word")
    if authored and hashes & JOB_WORD_HASHES:
        found.add("job-search wording")
    return found


def check_tree() -> list[str]:
    """Scan every tracked text file for banned content."""
    problems = []
    for name in sorted(tracked_files()):
        try:
            text = (ROOT / name).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        authored = not name.startswith(RAW_DATA)
        problems += [f"{name}: {found}" for found in sorted(text_problems(text, authored))]
    return problems


def fetch_problem(url: str) -> str | None:
    # No auth: every link has to open for a signed-out reader.
    request = urllib.request.Request(url, headers={"User-Agent": "launchready-check-site"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return None if response.status == 200 else f"{response.status}: {url}"
    except (urllib.error.URLError, TimeoutError) as error:
        return f"{error}: {url}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="fetch every external link")
    args = parser.parse_args()

    problems, external = check_site()
    problems += check_tree()
    if args.online:
        problems += [p for url in external if (p := fetch_problem(url))]
    for problem in problems:
        print(f"FAIL {problem}")
    mode = "fetched" if args.online else "checked offline"
    print(f"{len(external)} external links {mode}, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
