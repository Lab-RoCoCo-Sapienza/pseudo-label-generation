def modify_gt_mot(path, save_path):
    """
    Function to change the mot annotation file generated by CVAT annotation tool
    to be compliant with the TrackEval evaluation system

    Args:
        path (str): Path to the file to modify
        save_path (str): Path of the new corrected file

    Returns:
        the corrected file with the annotations
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for i, line in enumerate(lines):
            line = line.split(",")
            line[6] = -1
            line[7] = -1
            line[8] = -1
            line.append(-1)
            new_lines.append(line)
        f.close()

    with open(save_path, 'w') as f:
        for i, line in enumerate(new_lines):
            write_line = ','.join(str(e) for e in line)
            f.write(write_line + '\n')
        f.close()


if __name__ == "__main__":
    modify_gt_mot('./gt.txt', './new_gt.txt')
