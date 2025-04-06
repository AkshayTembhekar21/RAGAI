from atlassian import Confluence
from bs4 import BeautifulSoup
import html


def fetch_confluence_pages(apiToken, space_key):
    confluence = Confluence(
        url='https://base.atlassian.net/wiki',
        username='mail_id',
        password=apiToken,
        cloud=True
    )

    pages_data = []

    try:
        pages = confluence.get_all_pages_from_space(space=space_key, limit=10)
        for page in pages:
            page_id = page['id']
            title = page['title']
            content = confluence.get_page_by_id(page_id, expand='body.storage')
            html_content = content['body']['storage']['value']

            # Convert HTML to plain text
            soup = BeautifulSoup(html_content, 'html.parser')
            plain_text = soup.get_text()
            decoded_text = html.unescape(plain_text)

            pages_data.append({
                "title": title,
                "text": decoded_text
            })

    except Exception as e:
        print("❌ Error while fetching pages:")
        print(e)

    return pages_data
