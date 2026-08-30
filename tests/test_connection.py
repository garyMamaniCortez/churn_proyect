from churn_detection.data.connection import DatabaseConnectionFactory, PostgresConnectionSettings


def test_sqlalchemy_url_is_built_correctly():
    settings = PostgresConnectionSettings(
        host="db.internal",
        port=5432,
        database="gym_db",
        user="gym_user",
        password="s3cret",
    )

    assert settings.sqlalchemy_url == (
        "postgresql+psycopg2://gym_user:s3cret@db.internal:5432/gym_db"
    )


def test_create_engine_returns_engine_without_connecting():
    settings = PostgresConnectionSettings(
        host="db.internal",
        port=5432,
        database="gym_db",
        user="gym_user",
        password="s3cret",
    )

    engine = DatabaseConnectionFactory.create_engine(settings)

    # create_engine() is lazy: no network call happens until first .connect().
    assert str(engine.url) == "postgresql+psycopg2://gym_user:***@db.internal:5432/gym_db"
