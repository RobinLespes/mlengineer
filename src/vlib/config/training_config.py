FEATURES = ["season", "holiday", "workingday", "weather", "weekday", "month", "year", "hour", "temp", "humidity",
            "windspeed", "atemp"]
TARGET = 'count'
RF_PARAMS = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}
