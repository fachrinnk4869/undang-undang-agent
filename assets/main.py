# download many pdf
import os
import requests


def download_pdf(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        # cek apkaah filenya error
        if response.content.startswith(b'%PDF'):
            print(f"Downloaded successfully: {save_path}")
        else:
            print(f"Downloaded but file is not a valid PDF: {save_path}")
            # os.remove(save_path)  # Remove invalid file
            return
        with open(save_path, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    urls = [
        "https://peraturan.go.id/files/UUD1945.pdf",
    ]
    # for j in range(1945, 2026):
    for j in range(1968, 1967, -1):
        # for j in range(2025, 1944, -1):
        for i in range(1, 100):
            # urls.append(f"https://peraturan.go.id/files/uu{i}-{j}.pdf")
            # urls.append(f"https://peraturan.go.id/files/uu{i}-{j}bt.pdf")
            # urls.append(f"https://peraturan.go.id/files/uu{i}-{j}pjl.pdf")
            urls.append(f"https://peraturan.go.id/files/UU+NO+{i}+TH+{j}.pdf")
            urls.append(
                f"https://peraturan.go.id/files/UU+Nomor+{i}+Tahun+{j}.pdf")
            # check leading zero
            # i_str = str(i).zfill(2)
            # urls.append(f"https://peraturan.go.id/files/UU0{i_str}{j}.pdf")
            # urls.append(f"https://peraturan.go.id/files/uu0{i_str}{j}.pdf")
            # urls.append(
            #     f"https://peraturan.go.id/files/uu-no-{i}-tahun-{j}.pdf")
            print(f"Added URL: uu{i}-{j}.pdf")

    save_dir = "scrap"
    os.makedirs(save_dir, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        save_path = os.path.join(save_dir, filename)
        if not os.path.exists(save_path):
            download_pdf(url, save_path)
        else:
            print(f"File already exists: {save_path}, skipping download.")
