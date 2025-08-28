from pathlib import Path
import xml.etree.ElementTree as ET


def get_loc_values(xml_file_path: str) -> list[str]:
    path = Path(xml_file_path)
    if not path.exists() or not path.is_file():
        return []

    # Parse XML and collect all <loc> element texts in document order
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    loc_values: list[str] = []
    # Handle default namespace (e.g., sitemap schema) if present
    namespace_uri: str | None = None
    if root.tag.startswith("{"):
        end_brace = root.tag.find("}")
        if end_brace != -1:
            namespace_uri = root.tag[1:end_brace]

    if namespace_uri:
        loc_elems = root.findall(f".//{{{namespace_uri}}}loc")
    else:
        loc_elems = root.findall(".//loc")

    for loc in loc_elems:
        text = (loc.text or "").strip()
        if text:
            loc_values.append(text)
    return loc_values


def get_page_urls(xml_file_path: str, contents: list[str]) -> list[str]:
    matches: list[str] = []
    for value in get_loc_values(xml_file_path):
        if any(fragment in value for fragment in contents):
            matches.append(value)
    return matches


