def prediction(stock, n_days):
    import yfinance as yf
    import plotly.graph_objs as go
    # model
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    # from keras.models import Sequential
    from tensorflow.python.keras.models import Sequential
    # from keras.layers import Dense, LSTM
    from tensorflow.python.keras.layers import Dense, LSTM
    from datetime import date, timedelta


    # load the data
    df = yf.download(stock, period='5y')
    data = df.filter(['Close'])
    dataset = data.values
    print(df)

    # training_size = math.ceil(len(dataset)*0.9)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data  # [:training_size, :]

    # rest data will be our testing dataset
    # test_data = scaled_data[training_size-60:, :]
    # Split the data into x_train and y_train datasets
    # i.e, independent and dependent training features
    x_train = []
    y_train = []
    for i in range(60, len(scaled_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=100, epochs=5)
    # create the testing dataset

    x_input = scaled_data[-60:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []

    n_steps = 60
    i = 0
    while (i < n_days):

        if (len(temp_input) > 60):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            ypred = model.predict(x_input, verbose=0)
            temp_input.extend(ypred[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(ypred.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            ypred = model.predict(x_input, verbose=0)
            temp_input.extend(ypred[0].tolist())
            lst_output.extend(ypred.tolist())
            i = i + 1

    dates = [date.today()]
    current = date.today()
    for i in range(n_days - 1):
        current += timedelta(days=1)
        dates.append(current)

    print(dates)
    lst_output = scaler.inverse_transform(lst_output)
    lst_output = np.array(lst_output)
    op = lst_output.ravel()
    print(op)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,  # np.array(ten_days).flatten(),
            y=op,
            mode='lines+markers',
            name='data'))
    fig.update_layout(
        title="Predicted Close Price of next " + str(n_days - 1) + " days",
        xaxis_title="Date",
        yaxis_title="Closed Price",
        # legend_title="Legend Title",
    )

    return fig
prediction("AAPL",5)