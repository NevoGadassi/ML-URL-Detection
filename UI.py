import tkinter as tk
import pred
import warnings
import re

# ignoring the warnings from the imports
warnings.filterwarnings("ignore")


def url_tokenizer(url):
    tokens = re.split('[://?&=._-]+', url)
    return tokens


def ip_address_presence(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
    if match:
        return -1
    else:
        return 1


class UrlClassificator:
    def __init__(self, master) -> None:
        self.master = master
        master.title("URL Classificator")

        self.entry = tk.Entry(master, width=50, justify="center")
        self.entry.pack(pady=10)

        self.button = tk.Button(master, text="Make Classification", command=self.print_input)
        self.button.pack()

        self.label = tk.Label(master, text="Prediction:")
        self.label.pack(pady=10)

        self.credits = tk.Label(master, text="Lior Vinman :: Nevo Gadassi :: Yoad Tamar")
        self.credits.pack(pady=10)

    def print_input(self) -> None:
        prediction = "White" if self.make_classification() == 0 else "Malicious"
        self.label.config(text=("Prediction: " + prediction))

    def make_classification(self) -> int:
        url = self.entry.get()

        # checking if it's just an IP, adding protocol
        if ip_address_presence(url):
            if not url.startswith("https://") and not url.startswith("http://"):
                url = "https://" + url

        # checking if url not starts with protocol or W-W-W
        elif not (url.startswith("http://") or url.startswith("https://") or url.startswith("www.")):
            url = "https://www." + url

        # checking if starts with W-W-W and not with protocol
        elif url.startswith("www."):
            url = "https://" + url

        # checking if there is protocol (http case) but not www
        elif (url.startswith("http://")) and not (
                url.startswith("http://www.") or url.startswith("https://www.")):
            url = "https://www." + url[7:]

        # checking if there is protocol (https case) but not www
        elif (url.startswith("https://")) and not (
                url.startswith("http://www.") or url.startswith("https://www.")):
            url = "https://www." + url[8:]

        # logging the url
        print("url: " + url)

        return int(pred.check_url(url))


if __name__ == "__main__":
    root = tk.Tk()
    app = UrlClassificator(root)
    root.mainloop()
