import time

class Timer:
    def __init__(self, round_number):
        self.start_time = 0
        self.elapsed_time = 0
        self.block_size = 0
        self.round_number = round_number
        self.dataset = []
        self.score = 0

    def start_timer(self, block_size):
        self.start_time = time.time()
        self.block_size = block_size

    def end_timer(self):
        self.elapsed_time = time.time() - self.start_time
        self.dataset.append((self.elapsed_time, self.block_size))

    def show_report(self):
        pass

    def finalize(self, score):
        self.score = score
        self.write_to_file()

    def write_to_file(self):
        f = open("./output/simpleMCTS.txt", "a")
        output_string = str(self.round_number) + ","
        for i in range(self.round_number):
            output_string += str(self.dataset[i][0]) + "," + str(self.dataset[i][1]) + ","

        output_string += str(self.score) + "\n"
        f.write(output_string)
        f.close()
