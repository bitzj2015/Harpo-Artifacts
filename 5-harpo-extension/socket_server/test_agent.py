import uuid
import app
from prettytable import PrettyTable
from time import sleep
from tqdm import tqdm
from random import randint

app.NUM_REWARDS = 3

TRIALS = 3

recv_urls = []
uuids = []
url_cats = []

print("Testing training pipeline...")

def generate_history(rewards):
    count = 1

    for reward in tqdm(rewards):
        url_uuid, url_cat, urls = app.obfuscation_url()
        recv_urls.append(urls[0])
        uuids.append(url_uuid)
        url_cats.append(url_cat)
        print("Requesting obfuscation URL {0}...".format(count))
        print("Returning reward...")
        app.maintain_int_seg_history_test(url_uuid, reward)
        count += 1


def print_results(rewards):
    url_table = PrettyTable()

    url_table.field_names = ["URL category", "Reward"]

    i = 0

    for url in recv_urls:
        url_table.add_row([url_cats[i], rewards[i]])
        i += 1

    print(url_table)

while __name__ == "__main__":
    trial_num = 0

    for i in range(TRIALS):
        rewards = [randint(0, 3) for i in range(app.NUM_REWARDS)]
        print("TRIAL {0}".format(trial_num))
        generate_history(rewards)
        print_results(rewards)
        recv_urls = []
        uuids = []
        trial_num += 1