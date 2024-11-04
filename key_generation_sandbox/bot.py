import datetime
from pathlib import Path
import telebot
import traceback
import key_generation_sandbox
import utils


with open(Path.home() / ".ssh" / "unclonix" / "unclonix_hash_bot.txt") as f:
    key = f.readline().strip()
bot = telebot.TeleBot(key)


@bot.message_handler(commands=['start', 'help'])
def manual(message):
    bot.send_message(message.from_user.id, "To get started, send label image")


@bot.message_handler(content_types=['photo', 'document'])
def handle_photo(message):
    print('message.photo =', message.photo)
    file_id = message.photo[-1].file_id
    print('file_id =', file_id)
    file_info = bot.get_file(file_id)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    name = "bot" + datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S%f') + ".jpg"
    with open("../input/" + name, 'wb') as new_file:
        new_file.write(downloaded_file)
    utils.set_total_time()
    try:
        the_hash = key_generation_sandbox.run(name)
        bot.send_message(message.from_user.id, "Image hash is " + the_hash)
    except:
        print(traceback.format_exc())


def run():
    print('Running bot')
    bot.polling(none_stop=True, interval=0)


if __name__ == '__main__':
    run()
