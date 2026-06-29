from playwright.sync_api import sync_playwright, expect
import time

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Access Dashboard
        page.goto("http://localhost:8501")
        time.sleep(5) # Wait for Streamlit to load

        # 2. Screenshot Overview
        page.screenshot(path="verification_overview.png")
        print("Captured overview screenshot.")

        # 3. Navigate to Risk Matrix
        page.get_by_text("风险矩阵").click()
        time.sleep(3)
        page.screenshot(path="verification_matrix.png")
        print("Captured risk matrix screenshot.")

        browser.close()

if __name__ == "__main__":
    verify_dashboard()
