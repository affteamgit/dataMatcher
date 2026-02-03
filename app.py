import streamlit as st
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

    term = re.sub(r"\s+(logo|provider|slot|games?|studios?|entertainment|software)$", "", term, flags=re.IGNORECASE)
    term = re.sub(r"^(game\s+|slot\s+|casino\s+)", "", term, flags=re.IGNORECASE)
    term = re.sub(r"\s*[-–]\s*.*$", "", term)
    term = re.sub(r"\s+games?$", "", term, flags=re.IGNORECASE)
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


def call_claude(prompt, api_key):
    """Call Claude API with retry logic"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}]
    }

    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries + 1):
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            response.raise_for_status()
            return response.json().get("content", [{}])[0].get("text", "")
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
    """Use AI to match remaining terms"""
    PROMPTS = {
        "country": "From the casino terms: {terms}, which ones match these countries or their variations from the spreadsheet Seznami tab: {known}?",
        "language": "Which of these languages: {terms} match the languages from the spreadsheet Seznami tab: {known}?",
        "crypto": "Identify which terms in: {terms} match the cryptocurrency names or variations from the spreadsheet Seznami tab: {known}.",
        "provider": "Match game providers in: {terms} to the game providers and their variations in the spreadsheet Seznami tab: {known}."
    }

    seen = set()
    known_sample = ", ".join(list(known_dict.keys())[:100])
    input_text = ", ".join(terms)
    prompt = PROMPTS[category].format(terms=input_text, known=known_sample)

    claude_answer = call_claude(prompt, api_key)

    matched_terms = []
    results = []

    for term in terms:
        term_lc = term.lower()
        if term_lc in seen:
            continue
        if re.search(rf"\b{re.escape(term)}\b", claude_answer, re.IGNORECASE):
            if term_lc in known_dict:
                best_match = known_dict[term_lc]
                results.append({
                    "Element Type": category,
                    "Detected Term": term,
                    "Matched To": best_match[0],
                    "WP ID": best_match[1],
                    "Matched By": "AI"
                })
                seen.add(term_lc)
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
        if f"{cat}_input" not in st.session_state:
            st.session_state[f"{cat}_input"] = ""
        if f"{cat}_expanded" not in st.session_state:
            st.session_state[f"{cat}_expanded"] = False

    if "results" not in st.session_state:
        st.session_state.results = {}
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Logo in top left corner
    st.image("bitstarz-logo.svg", width=150)

    # Centered layout
    col1, center, col2 = st.columns([1, 2, 1])

    with center:
        st.markdown("<h2 style='text-align: center; margin-bottom: 1rem;'>Data Matcher</h2>", unsafe_allow_html=True)

        # Input sections - compact expanders
        for cat_key, cat_display in CATEGORIES.items():
            with st.expander(f"{cat_display}", expanded=st.session_state[f"{cat_key}_expanded"]):
                st.session_state[f"{cat_key}_input"] = st.text_area(
                    f"Paste {cat_display} HTML/text here:",
                    value=st.session_state[f"{cat_key}_input"],
                    height=80,
                    key=f"{cat_key}_textarea",
                    label_visibility="collapsed",
                    placeholder=f"Paste HTML or comma-separated list..."
                )
                if st.button(f"Clear", key=f"clear_{cat_key}", use_container_width=True):
                    st.session_state[f"{cat_key}_input"] = ""
                    st.rerun()

        st.markdown("")
        match_button = st.button("MATCH", type="primary", use_container_width=True)

    if match_button:
        # Check if any input provided
        has_input = any(st.session_state[f"{cat}_input"].strip() for cat in CATEGORIES.keys())

        if not has_input:
            st.warning("Please provide input for at least one category.")
        else:
            st.session_state.processing = True
            st.session_state.results = {}

            # Load sheet data
            with st.spinner("Loading database..."):
                try:
                    known_data = load_sheet_data(creds)
                except Exception as e:
                    st.error(f"Failed to load database: {e}")
                    st.session_state.processing = False
                    st.stop()

            all_matched = []
            progress_container = st.container()

            # Process each category
            for cat_key, cat_display in CATEGORIES.items():
                input_content = st.session_state[f"{cat_key}_input"].strip()

                if not input_content:
                    continue

                with progress_container:
                    status = st.status(f"Processing {cat_display}...", expanded=True)

                    with status:
                        st.write("Extracting terms...")

                        def progress_cb(msg):
                            st.write(msg)

                        terms = process_html_input(input_content, cat_key, api_key, progress_cb)
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

                        status.update(label=f"{cat_display} complete!", state="complete")

            # Filter results
            if all_matched:
                verified, rejected = automated_proof_filter(all_matched)
                st.session_state.results = generate_results(verified)

            st.session_state.processing = False

            # Collapse input sections after processing
            for cat in CATEGORIES.keys():
                st.session_state[f"{cat}_expanded"] = False

            st.rerun()

    # Display results (centered)
    if st.session_state.results:
        col1, center_results, col2 = st.columns([1, 2, 1])
        with center_results:
            st.markdown("---")
            st.markdown("<h4 style='text-align: center;'>Results (Ready to Copy)</h4>", unsafe_allow_html=True)

            for cat_key, cat_display in CATEGORIES.items():
                if cat_key in st.session_state.results:
                    result_text = st.session_state.results[cat_key]
                    st.markdown(f"**{cat_display}:**")
                    st.code(result_text, language=None)

            if st.button("Clear All", use_container_width=True):
                st.session_state.results = {}
                for cat in CATEGORIES.keys():
                    st.session_state[f"{cat}_input"] = ""
                    st.session_state[f"{cat}_expanded"] = False
                st.rerun()


if __name__ == "__main__":
    main()
