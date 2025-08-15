'''
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
'''
# rename file pdf ke satu jenis uu{i}-{j}
import os
import re
import shutil


def rename_pdf_files(directory):
    for filename in os.listdir(directory):
        # if filename.endswith(".pdf"):

        # Extract the year and number from the filename
        match = re.search(r'uu(\d+)-(\d{4})', filename, re.IGNORECASE)
        if not match:
            match = re.search(
                r'UU\+NO\+(\d+)\+TH\+(\d{4})', filename, re.IGNORECASE)
        if not match:
            match = re.search(
                r'UU\+Nomor\+(\d+)\+Tahun\+(\d{4})', filename, re.IGNORECASE)
        if not match:
            match = re.search(
                r'uu-no-(\d+)-tahun-(\d{4})', filename, re.IGNORECASE)
        if not match:
            match = re.search(
                r'uu0(\d{2})(\d{4})', filename, re.IGNORECASE)
        if match:
            print(f"Processing file: {filename}")
            number, year = match.groups()
            new_filename = f"uu{int(number)}-{year}.pdf"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # if not os.path.exists(new_path):
            # its not removed old path
            shutil.move(old_path, new_path)
            #
            print(f"Renamed {filename} to {new_filename}")
            # else:
            #     print(
            #         f"File already exists: {new_filename}, skipping rename.")
        # else:
        #     print(f"No match found for {filename}, skipping.")
        #     # if filename == "uu1-2014pjl.pdf":
        #     #     break


if __name__ == "__main__":
    directory = "scrap (copy)"
    rename_pdf_files(directory)
