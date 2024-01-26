
def performance_evaluation(match_L1, match_L2, match_cosine):
    # Extract correctly matched elements
    matching_L1 = [i for i in match_L1 if i == 1]
    matching_L2 = [i for i in match_L2 if i == 1]
    matching_cosine = [i for i in match_cosine if i == 1]

    # Calculate correct recognition rates
    recognition_L1 = len(matching_L1) / len(match_L1)
    recognition_L2 = len(matching_L2) / len(match_L2)
    recognition_cosine = len(matching_cosine) / len(match_cosine)

    return recognition_L1 * 100, recognition_L2 * 100, recognition_cosine * 100
