import streamlit as st
import base64
import html
import re
import time
import random
import requests
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Page config
st.set_page_config(
    page_title="Data Matcher",
    page_icon="🎰",
    layout="wide"
)

# Force dark theme with custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }

    /* Text color */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #fafafa;
    }

    /* Expander background */
    [data-testid="stExpander"] {
        background-color: #262730;
        border-radius: 8px;
    }

    /* Text area background */
    [data-testid="stTextArea"] textarea {
        background-color: #262730;
        color: #fafafa;
    }

    /* Code block background */
    [data-testid="stCode"] {
        background-color: #262730;
    }

    /* Input fields */
    .stTextInput input, .stSelectbox select {
        background-color: #262730;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Google Sheets Info
SPREADSHEET_ID = "1ZneRUz90Ne06pr8CCax8vp30tOtPpKJQCw5ikE-uB_0"
SHEET_NAME = "Lists"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Category display names
CATEGORIES = {
    "crypto": "Cryptocurrencies",
    "country": "Countries",
    "language": "Languages",
    "provider": "Game Providers"
}

# FATF "High-Risk Jurisdictions subject to a Call for Action" (the FATF blacklist).
# Casinos often write "FATF blacklisted countries" instead of naming them. Update
# this list by hand whenever FATF revises it (see fatf-gafi.org) - last checked 2026.
FATF_BLACKLIST_COUNTRIES = ["Iran", "North Korea", "Myanmar"]

FATF_MENTION_RE = re.compile(r"\bfatf\b", re.IGNORECASE)


def get_anthropic_api_key():
    """Get Anthropic API key from Streamlit secrets"""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


def get_google_credentials():
    """Get Google credentials from Streamlit secrets"""
    try:
        creds_dict = st.secrets["service_account"]
        creds = Credentials.from_service_account_info(dict(creds_dict), scopes=SCOPES)
        return creds
    except Exception as e:
        st.error(f"Failed to load Google credentials: {e}")
        return None


def clean_term(term):
    """Clean and normalize a term"""
    if not term:
        return None

    term = html.unescape(term)
    term = re.sub(r"^\d+\.\s*", "", term)
    term = re.sub(r"\(.*?\)", "", term)
    term = re.sub(r"<[^>]*>", "", term)
    term = re.sub(r'[\"""'']+', "", term)
    term = term.strip()

    term = re.sub(r"\s+(logo|provider|slot|studios?|entertainment|software)$", "", term, flags=re.IGNORECASE)
    term = re.sub(r"^(game\s+|slot\s+|casino\s+)", "", term, flags=re.IGNORECASE)
    term = re.sub(r"\s*[-–]\s*.*$", "", term)
    term = re.sub(r"\s+", " ", term)
    term = term.strip()

    if not term or len(term) < 2:
        return None
    if term.isdigit():
        return None
    if re.match(r"^\d+\s*$", term):
        return None
    if len(term) == 1:
        return None
    if term.lower() in {"all studios", "all providers", "studios", "providers", "games", "slots", "casino", "gaming", "b", "g"}:
        return None

    return term


def chunk_html(html_content, chunk_size=8000):
    """Split HTML content into overlapping chunks"""
    chunks = []
    for i in range(0, len(html_content), chunk_size):
        chunk = html_content[i:i + chunk_size + 1000]
        chunks.append(chunk)
    return chunks


def call_claude(content, api_key):
    """Call Claude API with retry logic. `content` is either a text prompt (str)
    or a list of content blocks (e.g. images + text) for vision requests."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 4000,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": content}]
    }

    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries + 1):
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            response.raise_for_status()
            content_blocks = response.json().get("content", [])
            return next((block.get("text", "") for block in content_blocks if block.get("type") == "text"), "")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [429, 529]:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
            return ""
        except Exception:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            return ""
    return ""


def extract_providers_with_regex(html_content):
    """Extract provider names using regex patterns"""
    providers = set()

    alt_pattern = r'alt="([^"]+)"'
    alt_matches = re.findall(alt_pattern, html_content, re.IGNORECASE)
    for match in alt_matches:
        cleaned = clean_term(match)
        if cleaned and len(cleaned) > 2:
            providers.add(cleaned)

    href_pattern = r'providerName=([^&"]+)'
    href_matches = re.findall(href_pattern, html_content, re.IGNORECASE)
    for match in href_matches:
        cleaned = clean_term(match.replace('%20', ' '))
        if cleaned and len(cleaned) > 2:
            providers.add(cleaned)

    data_pattern = r'data-provider[^=]*="([^"]+)"'
    data_matches = re.findall(data_pattern, html_content, re.IGNORECASE)
    for match in data_matches:
        cleaned = clean_term(match)
        if cleaned and len(cleaned) > 2:
            providers.add(cleaned)

    return list(providers)


def extract_terms_with_ai(html_content, category, api_key, progress_callback=None):
    """Extract terms from HTML using AI"""

    enhanced_prompts = {
        "provider": """
Extract game provider/software company names from this HTML.

Focus on:
- img alt attributes (like alt="Thunderkick", alt="NetEnt")
- href URLs with /providers/ or /games/ paths
- Company names in text content

Extract ONLY actual provider names like: Pragmatic Play, NetEnt, Microgaming, Evolution, Thunderkick, BGaming, Yggdrasil.

DO NOT extract: games, logo, provider, software, icon, image.

If no providers found, return: NONE

HTML:
{html_content}

Provider names (one per line):""",

        "country": """
You are extracting country names from casino website HTML.

INSTRUCTIONS:
1. Look for country names in text, alt attributes, and data attributes
2. Check for flag images with country names in alt text
3. Look for country-specific content sections

Extract only actual country names, not generic terms.

HTML content to analyze:
{html_content}

Return each country name on a separate line, nothing else:""",

        "language": """
You are extracting language names from casino website HTML.

INSTRUCTIONS:
1. Look for language names in text, alt attributes, and lang attributes
2. Check for language selection menus or buttons
3. Look for multilingual content indicators

Extract only actual language names.

HTML content to analyze:
{html_content}

Return each language name on a separate line, nothing else:""",

        "crypto": """
You are extracting cryptocurrency and payment method names from casino website HTML.

INSTRUCTIONS:
1. Look for crypto names in text, alt attributes, and payment sections
2. Check for payment method lists and crypto wallet options
3. Look for blockchain-related terms

Extract cryptocurrency names and payment methods.

HTML content to analyze:
{html_content}

Return each cryptocurrency/payment method on a separate line, nothing else:"""
    }

    if category == "provider" or len(html_content) > 8000:
        chunks = chunk_html(html_content, chunk_size=6000)
        all_extracted_terms = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(f"Processing chunk {i+1}/{len(chunks)}...")

            prompt = enhanced_prompts[category].format(html_content=chunk)
            claude_result = call_claude(prompt, api_key)

            if claude_result:
                chunk_terms = [term.strip() for term in claude_result.split('\n') if term.strip() and term.strip().upper() != 'NONE']
                chunk_terms = [term for term in chunk_terms if not any(phrase in term.lower() for phrase in [
                    'i don\'t see', 'there are no', 'appears to be', 'snippet', 'html', 'based on',
                    'these providers', 'found in', 'urls', 'attributes', 'href', 'alt', 'images'
                ]) and len(term) < 100]
                all_extracted_terms.extend(chunk_terms)

        extracted_terms = all_extracted_terms
    else:
        prompt = enhanced_prompts[category].format(html_content=html_content)

        extracted_terms = []
        claude_result = call_claude(prompt, api_key)
        if claude_result:
            extracted_terms = [term.strip() for term in claude_result.split('\n') if term.strip() and term.strip().upper() != 'NONE']
            extracted_terms = [term for term in extracted_terms if not any(phrase in term.lower() for phrase in [
                'i don\'t see', 'there are no', 'appears to be', 'snippet', 'html', 'based on',
                'these providers', 'found in', 'urls', 'attributes', 'href', 'alt', 'images'
            ]) and len(term) < 100]

    cleaned_terms = []
    seen_lower = set()
    for term in extracted_terms:
        cleaned = clean_term(term)
        if cleaned:
            cleaned_lower = cleaned.lower()
            if cleaned_lower not in seen_lower:
                cleaned_terms.append(cleaned)
                seen_lower.add(cleaned_lower)

    return cleaned_terms


def extract_terms_from_images(image_files, category, api_key):
    """Extract terms from screenshot(s) using Claude vision.

    Returns (terms, unreadable_notes) - unreadable_notes are tiles whose logo
    has no legible name and wasn't confidently identifiable, so they're
    surfaced for manual review instead of guessed.
    """
    content_blocks = []
    for image_file in image_files:
        media_type = image_file.type or "image/png"
        image_b64 = base64.standard_b64encode(image_file.getvalue()).decode("utf-8")
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_b64}
        })

    label = CATEGORIES[category].lower()
    prompt = f"""These screenshot(s) show a grid or list of {label} logos/icons from a casino website.

For EACH distinct logo or tile, output exactly one line:
- If the logo contains a legible brand name or wordmark, output just that name.
- If the logo is a pure icon/symbol with no legible name and you cannot identify it with high confidence, output: UNREADABLE: <short visual description, e.g. colors and shape>

Do not guess a name for a logo you cannot confidently identify - use the UNREADABLE line instead. Ignore UI elements that aren't {label} (search boxes, navigation, "All"/"All providers" tiles, close buttons, etc.). Output one line per tile, nothing else."""

    content_blocks.append({"type": "text", "text": prompt})

    claude_answer = call_claude(content_blocks, api_key)

    terms = []
    unreadable = []

    if claude_answer:
        for line in claude_answer.splitlines():
            line = line.strip().lstrip("-").strip()
            if not line:
                continue
            if line.upper().startswith("UNREADABLE:"):
                description = line.split(":", 1)[1].strip().replace(",", ";")
                unreadable.append(f"Unreadable logo: {description}" if description else "Unreadable logo")
            else:
                cleaned = clean_term(line)
                if cleaned:
                    terms.append(cleaned)

    seen_lower = set()
    unique_terms = []
    for term in terms:
        term_lower = term.lower()
        if term_lower not in seen_lower:
            seen_lower.add(term_lower)
            unique_terms.append(term)

    return unique_terms, unreadable


def expand_fatf_mentions(raw_input, terms):
    """If a country input mentions FATF (e.g. "FATF blacklisted countries"),
    drop that mention and expand it into the individual FATF-blacklisted
    countries so they go through the normal matching pipeline."""
    mentions_fatf = bool(FATF_MENTION_RE.search(raw_input or "")) or any(
        FATF_MENTION_RE.search(term) for term in terms
    )
    if not mentions_fatf:
        return terms

    expanded = [term for term in terms if not FATF_MENTION_RE.search(term)]
    seen_lower = {t.lower() for t in expanded}
    for country in FATF_BLACKLIST_COUNTRIES:
        if country.lower() not in seen_lower:
            expanded.append(country)
            seen_lower.add(country.lower())
    return expanded


def process_html_input(html_content, category, api_key, progress_callback=None):
    """Process HTML content using AI extraction or direct parsing"""

    if '<' not in html_content and ',' in html_content:
        terms = []
        for term in html_content.split(','):
            cleaned = clean_term(term.strip())
            if cleaned:
                terms.append(cleaned)

        seen = set()
        unique_terms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)

        return unique_terms

    if category == "provider":
        regex_terms = extract_providers_with_regex(html_content)
        ai_terms = extract_terms_with_ai(html_content, category, api_key, progress_callback)

        final_terms = []
        seen_lower = set()

        for term in regex_terms:
            term_lower = term.lower()
            if term_lower not in seen_lower:
                final_terms.append(term)
                seen_lower.add(term_lower)

        for term in ai_terms:
            term_lower = term.lower()
            if term_lower not in seen_lower:
                final_terms.append(term)
                seen_lower.add(term_lower)

        return final_terms

    return extract_terms_with_ai(html_content, category, api_key, progress_callback)


@st.cache_data(ttl=300)
def load_sheet_data(_creds):
    """Load data from Google Sheets"""
    service = build("sheets", "v4", credentials=_creds)
    range_str = f"{SHEET_NAME}!A2:L"
    rows = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=range_str).execute().get("values", [])

    categories = {
        "country": (0, 1, 2),
        "language": (3, 4, 5),
        "crypto": (6, 7, 8),
        "provider": (9, 10, 11)
    }

    parsed = {}
    for cat, (main_i, var_i, wp_i) in categories.items():
        known = {}
        for row in rows:
            if len(row) > main_i and row[main_i].strip():
                main = row[main_i].strip()
                wp_id = row[wp_i].strip() if len(row) > wp_i else ""

                main_clean = clean_term(main)
                if main_clean:
                    known[main_clean.lower()] = (main, wp_id)

                if len(row) > var_i and row[var_i].strip():
                    variations = row[var_i].split(",")
                    for var in variations:
                        var_clean = clean_term(var)
                        if var_clean:
                            # Only check for duplicate if main_clean exists
                            if not main_clean or var_clean.lower() != main_clean.lower():
                                known[var_clean.lower()] = (main, wp_id)

        parsed[cat] = known
    return parsed


def fuzzy_match(term, known_dict, threshold=85):
    """Returns the best fuzzy match key from known_dict if ratio >= threshold"""
    if not known_dict:
        return None
    match, score = process.extractOne(term, known_dict.keys(), scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return match
    return None


def match_elements(category, input_list, known_dict):
    """Match extracted terms against known database"""
    seen = set()
    matched, unmatched = [], []

    for term in input_list:
        term_lc = term.lower()
        if term_lc in seen:
            continue

        match_key = term_lc if term_lc in known_dict else fuzzy_match(term, known_dict, threshold=92)

        if match_key and match_key not in seen:
            seen.add(match_key)
            matched.append({
                "Element Type": category,
                "Detected Term": term,
                "Matched To": known_dict[match_key][0],
                "WP ID": known_dict[match_key][1],
                "Matched By": "exact" if match_key == term_lc else "fuzzy"
            })
        else:
            unmatched.append(term)

    return matched, unmatched


def ai_match_terms(category, terms, known_dict, api_key):
    """Use AI to match remaining terms against the full known list"""
    LABELS = {
        "country": "countries",
        "language": "languages",
        "crypto": "cryptocurrencies",
        "provider": "game providers"
    }

    # Build the canonical name list (deduped) and a name -> WP ID lookup
    name_to_wpid = {}
    for main, wp_id in known_dict.values():
        name_to_wpid.setdefault(main, wp_id)
    known_names = sorted(name_to_wpid.keys())
    known_lower_to_name = {name.lower(): name for name in known_names}

    input_lines = "\n".join(f"- {term}" for term in terms)
    known_text = ", ".join(known_names)

    prompt = f"""You are matching casino industry terms to a known list of {LABELS[category]}.

Input terms (may be misspelled, abbreviated, or formatted differently):
{input_lines}

Known list of correct {LABELS[category]} names:
{known_text}

For EACH input term above, respond with exactly one line in this format:
<input term> => <matching name from the known list>

If an input term does not match anything in the known list, respond with:
<input term> => NONE

Use the exact spelling from the known list on the right side. Do not invent names that aren't in the list. Respond with nothing else."""

    claude_answer = call_claude(prompt, api_key)

    matched_terms = []
    results = []

    if claude_answer:
        for line in claude_answer.splitlines():
            if "=>" not in line:
                continue
            left, _, right = line.partition("=>")
            left = left.strip().lstrip("-").strip()
            right = right.strip()

            term = next((t for t in terms if t.lower() == left.lower()), None)
            if not term or term in matched_terms or not right or right.upper() == "NONE":
                continue

            canonical = known_lower_to_name.get(right.lower())
            if not canonical:
                fuzzy = process.extractOne(right, known_names, scorer=fuzz.token_sort_ratio)
                if fuzzy and fuzzy[1] >= 90:
                    canonical = fuzzy[0]

            if canonical:
                results.append({
                    "Element Type": category,
                    "Detected Term": term,
                    "Matched To": canonical,
                    "WP ID": name_to_wpid.get(canonical, ""),
                    "Matched By": "AI"
                })
                matched_terms.append(term)

    unmatched = [t for t in terms if t not in matched_terms]
    return results, unmatched


def automated_proof_filter(matched_list):
    """Apply automated filters to reduce false positives"""
    auto_approved = []
    auto_rejected = []

    for match in matched_list:
        detected = match['Detected Term'].lower()
        matched_to = match['Matched To'].lower()
        match_type = match['Matched By']

        if match_type == "exact":
            auto_approved.append(match)
            continue

        if match_type == "fuzzy":
            similarity = fuzz.ratio(detected, matched_to)
            if similarity >= 95:
                auto_approved.append(match)
                continue

        len_ratio = min(len(detected), len(matched_to)) / max(len(detected), len(matched_to))
        if len_ratio < 0.3:
            auto_rejected.append(match)
            continue

        detected_words = set(detected.split())
        matched_words = set(matched_to.split())
        if len(detected_words) > 1 and len(matched_words) > 1:
            if not detected_words.intersection(matched_words):
                auto_rejected.append(match)
                continue

        auto_approved.append(match)

    return auto_approved, auto_rejected


def generate_results(matched_list):
    """Generate ready-to-copy results grouped by category"""
    category_entities = {}

    for match in matched_list:
        category = match['Element Type']
        entity_name = match['Matched To']

        if entity_name:
            if category not in category_entities:
                category_entities[category] = set()
            category_entities[category].add(entity_name)

    results = {}
    for category, entities_set in category_entities.items():
        if entities_set:
            entities_list = sorted(list(entities_set))
            results[category] = ", ".join(entities_list)

    return results


def clear_category_input(cat_key):
    """Reset one category's input. Must run via on_click - session_state for a
    widget's own key can't be set after that widget has rendered this run."""
    st.session_state[f"{cat_key}_textarea"] = ""
    st.session_state[f"{cat_key}_uploader_version"] += 1


def clear_all_results():
    """Reset all results and inputs. Must run via on_click (see clear_category_input)."""
    st.session_state.results = {}
    st.session_state.unmatched = {}
    st.session_state.ai_matches = {}
    for cat in CATEGORIES.keys():
        st.session_state[f"{cat}_textarea"] = ""
        st.session_state[f"{cat}_expanded"] = False
        st.session_state[f"{cat}_uploader_version"] += 1


def main():
    # Check for API credentials first
    api_key = get_anthropic_api_key()
    creds = get_google_credentials()

    if not api_key:
        st.error("Anthropic API key not found. Please add ANTHROPIC_API_KEY to your Streamlit secrets.")
        st.stop()

    if not creds:
        st.error("Google credentials not found. Please add service_account to your Streamlit secrets.")
        st.stop()

    # Initialize session state for inputs
    for cat in CATEGORIES.keys():
        if f"{cat}_textarea" not in st.session_state:
            st.session_state[f"{cat}_textarea"] = ""
        if f"{cat}_expanded" not in st.session_state:
            st.session_state[f"{cat}_expanded"] = False
        if f"{cat}_uploader_version" not in st.session_state:
            st.session_state[f"{cat}_uploader_version"] = 0

    if "results" not in st.session_state:
        st.session_state.results = {}
    if "unmatched" not in st.session_state:
        st.session_state.unmatched = {}
    if "ai_matches" not in st.session_state:
        st.session_state.ai_matches = {}
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Logo in top left corner
    st.image("bitstarz-logo.svg", width=150)

    # Centered layout
    col1, center, col2 = st.columns([1, 2, 1])

    with center:
        st.markdown("<h2 style='text-align: center; margin-bottom: 1rem;'>Data Matcher</h2>", unsafe_allow_html=True)

        # Input sections - compact expanders
        category_images = {}
        for cat_key, cat_display in CATEGORIES.items():
            with st.expander(f"{cat_display}", expanded=st.session_state[f"{cat_key}_expanded"]):
                st.text_area(
                    f"Paste {cat_display} HTML/text here:",
                    height=80,
                    key=f"{cat_key}_textarea",
                    label_visibility="collapsed",
                    placeholder=f"Paste HTML or comma-separated list..."
                )
                category_images[cat_key] = st.file_uploader(
                    f"Or upload screenshot(s) of {cat_display.lower()} logos/icons",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key=f"{cat_key}_images_uploader_v{st.session_state[f'{cat_key}_uploader_version']}"
                )
                st.button(
                    f"Clear", key=f"clear_{cat_key}", use_container_width=True,
                    on_click=clear_category_input, args=(cat_key,)
                )

        st.markdown("")
        match_button = st.button("MATCH", type="primary", use_container_width=True)

    if match_button:
        # Check if any input provided
        has_input = any(
            st.session_state[f"{cat}_textarea"].strip() or category_images.get(cat)
            for cat in CATEGORIES.keys()
        )

        if not has_input:
            st.warning("Please provide input for at least one category.")
        else:
            st.session_state.processing = True
            st.session_state.results = {}
            st.session_state.unmatched = {}
            st.session_state.ai_matches = {}

            # Load sheet data
            with st.spinner("Loading database..."):
                try:
                    known_data = load_sheet_data(creds)
                except Exception as e:
                    st.error(f"Failed to load database: {e}")
                    st.session_state.processing = False
                    st.stop()

            all_matched = []
            unmatched_terms = {}
            progress_container = st.container()

            # Process each category
            for cat_key, cat_display in CATEGORIES.items():
                input_content = st.session_state[f"{cat_key}_textarea"].strip()
                images = category_images.get(cat_key) or []

                if not input_content and not images:
                    continue

                with progress_container:
                    status = st.status(f"Processing {cat_display}...", expanded=True)

                    with status:
                        st.write("Extracting terms...")

                        def progress_cb(msg):
                            st.write(msg)

                        terms = process_html_input(input_content, cat_key, api_key, progress_cb) if input_content else []

                        if images:
                            st.write(f"Analyzing {len(images)} screenshot(s)...")
                            image_terms, unreadable_logos = extract_terms_from_images(images, cat_key, api_key)

                            seen_lower = {t.lower() for t in terms}
                            for term in image_terms:
                                if term.lower() not in seen_lower:
                                    terms.append(term)
                                    seen_lower.add(term.lower())

                            if unreadable_logos:
                                unmatched_terms.setdefault(cat_key, []).extend(unreadable_logos)

                            st.write(f"Screenshot(s): {len(image_terms)} name(s) read, {len(unreadable_logos)} unreadable logo(s)")

                        if cat_key == "country":
                            expanded_terms = expand_fatf_mentions(input_content, terms)
                            if len(expanded_terms) != len(terms):
                                st.write("Detected FATF mention - expanded to individual blacklisted countries")
                            terms = expanded_terms

                        st.write(f"Found {len(terms)} terms")

                        if terms:
                            st.write("Matching against database...")
                            base_match, remaining = match_elements(cat_key, terms, known_data[cat_key])
                            all_matched.extend(base_match)
                            st.write(f"Direct matches: {len(base_match)}")

                            if remaining:
                                st.write(f"Processing {len(remaining)} unmatched terms with AI...")
                                ai_matched, still_unmatched = ai_match_terms(cat_key, remaining, known_data[cat_key], api_key)
                                all_matched.extend(ai_matched)
                                st.write(f"AI matches: {len(ai_matched)}")

                                if still_unmatched:
                                    unmatched_terms.setdefault(cat_key, []).extend(still_unmatched)

                        status.update(label=f"{cat_display} complete!", state="complete")

            # Filter results
            ai_match_pairs = {}
            if all_matched:
                verified, rejected = automated_proof_filter(all_matched)
                st.session_state.results = generate_results(verified)

                for match in verified:
                    if match["Matched By"] == "AI":
                        ai_match_pairs.setdefault(match["Element Type"], set()).add(
                            (match["Detected Term"], match["Matched To"])
                        )

                for rej in rejected:
                    unmatched_terms.setdefault(rej["Element Type"], []).append(rej["Detected Term"])

            # Format AI-derived matches for verification (not exact/fuzzy matches against the known list)
            st.session_state.ai_matches = {}
            for cat_key, pairs in ai_match_pairs.items():
                sorted_pairs = sorted(pairs, key=lambda p: p[0].lower())
                st.session_state.ai_matches[cat_key] = "\n".join(f"{detected} → {matched}" for detected, matched in sorted_pairs)

            # Deduplicate and alphabetize unmatched terms per category
            st.session_state.unmatched = {}
            for cat_key, terms_list in unmatched_terms.items():
                seen_lower = set()
                unique_terms = []
                for term in terms_list:
                    term_lower = term.lower()
                    if term_lower not in seen_lower:
                        seen_lower.add(term_lower)
                        unique_terms.append(term)
                if unique_terms:
                    st.session_state.unmatched[cat_key] = ", ".join(sorted(unique_terms, key=str.lower))

            st.session_state.processing = False

            # Collapse input sections after processing
            for cat in CATEGORIES.keys():
                st.session_state[f"{cat}_expanded"] = False

            st.rerun()

    # Display results (centered)
    if st.session_state.results or st.session_state.unmatched or st.session_state.ai_matches:
        col1, center_results, col2 = st.columns([1, 2, 1])
        with center_results:
            st.markdown("---")

            if st.session_state.results:
                st.markdown("<h4 style='text-align: center;'>Results (Ready to Copy)</h4>", unsafe_allow_html=True)

                for cat_key, cat_display in CATEGORIES.items():
                    if cat_key in st.session_state.results:
                        result_text = st.session_state.results[cat_key]
                        st.markdown(f"**{cat_display}:**")
                        st.code(result_text, language=None)

            if st.session_state.ai_matches:
                st.markdown("<h4 style='text-align: center;'>AI-Matched (Please Verify)</h4>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: center; font-size: 0.85rem;'>"
                    "Matched by AI, not an exact/fuzzy hit against the known list &mdash; already included above. "
                    "Spot-check these, then add the detected spelling as a variation for the matched entity."
                    "</p>",
                    unsafe_allow_html=True
                )

                for cat_key, cat_display in CATEGORIES.items():
                    if cat_key in st.session_state.ai_matches:
                        st.markdown(f"**{cat_display}:**")
                        st.code(st.session_state.ai_matches[cat_key], language=None)

            if st.session_state.unmatched:
                st.markdown("<h4 style='text-align: center;'>Unmatched Terms</h4>", unsafe_allow_html=True)

                for cat_key, cat_display in CATEGORIES.items():
                    if cat_key in st.session_state.unmatched:
                        unmatched_text = st.session_state.unmatched[cat_key]
                        st.markdown(f"**{cat_display}:**")
                        st.code(unmatched_text, language=None)

            st.button("Clear All", use_container_width=True, on_click=clear_all_results)


if __name__ == "__main__":
    main()
