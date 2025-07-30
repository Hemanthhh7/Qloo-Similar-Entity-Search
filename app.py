import streamlit as st
import requests
import time
import re
from typing import Dict, List, Optional

# ✅ Hardcoded API Key
API_KEY = "CKyYlRKhl9HptdYL43cCTubBh75IqpWtc2ebbNO9MXM"

class QlooEntity:
    def __init__(self, name: str, raw_data: Optional[Dict] = None):
        self.name = name
        self.raw_data = raw_data or {}

    def __str__(self):
        return self.name

class QlooAPI:
    def __init__(self, api_key: str, base_url: str = "https://hackathon.api.qloo.com"):
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = base_url
        self.last_request_time = 0
        self.min_request_interval = 0.1

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def search(self, query: str, limit: int = 20) -> List[QlooEntity]:
        self._rate_limit()
        response = self.session.get(f"{self.base_url}/search", params={"query": query, "limit": limit})
        if response.status_code == 200:
            data = response.json()
            return [QlooEntity(name=item.get("name", "Unknown")) for item in data.get("results", [])]
        else:
            st.error(f"❌ API error: {response.status_code}")
            return []

# ✅ Utility function to normalize names
def normalize_name(name: str) -> str:
    return re.sub(r'\W+', '', name).lower().strip()

def main():
    st.set_page_config(page_title="Qloo Similar Name Search", layout="centered")
    st.title("🎬 Qloo Similar Entity Search")

    query = st.text_input("🔍 Enter a name (e.g., movie, artist, etc.):")

    if st.button("Search"):
        if not query:
            st.warning("Please enter a name to search.")
        else:
            api = QlooAPI(API_KEY)
            results = api.search(query)

            seen_normalized = set()
            unique_results = []

            for entity in results:
                norm = normalize_name(entity.name)
                key = (norm, len(norm))
                if key not in seen_normalized:
                    seen_normalized.add(key)
                    unique_results.append(entity)

            if unique_results:
                st.success(f"Found {len(unique_results)} unique similar names:")
                for i, entity in enumerate(unique_results, 1):
                    st.write(f"{i}. {entity.name}")
            else:
                st.info("No unique results found.")

if __name__ == "__main__":
    main()
