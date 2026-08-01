from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "klcantrell/so101_spam_musubi_place_spam_left_task" 
dataset = LeRobotDataset(repo_id)

print(dataset[0])

