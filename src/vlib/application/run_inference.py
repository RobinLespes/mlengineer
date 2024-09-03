from vlib.domain.utils import get_inference_data, append_predictions


def infer():
    url = 'http://127.0.0.1:5001/predict'
    next_date_inference, predictions = get_inference_data()

    # YOUR CODE HERE
    # you should infer the new column prediction to the nex_date_inference dataframe

    predictions = append_predictions(next_date_inference, predictions)
    predictions.to_csv("data/predictions.csv", index=False)


if __name__ == '__main__':
    infer()
