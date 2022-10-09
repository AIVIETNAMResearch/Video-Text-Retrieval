# pip install sqlalchemy
# pip install psycopg2-binary


import sqlalchemy as sa
import psycopg2

user = 'postgres'
password = 'hcmai'
dbms = 'postgresql'
db_lib = 'psycopg2'
host = '127.0.0.1'
port = 5432
database_name = 'VideoManager'

connect_str = f'{dbms}+{db_lib}://{user}:{password}@{host}:{port}/{database_name}'

engine = sa.create_engine(url=connect_str).connect()

print(engine)


"""
Folder orgnization:
-videos
    --VideoC00_V00
        ---C00_V0000.mp4
        ---C00_V0001.mp4
    --VideoC01_V00
        ---C01_V0000.mp4
        ---C01_V0001.mp4
    --VideoC02_V00
        ---C02_V0000.mp4
        ---C02_V0001.mp4
-keyframes
    --KeyFramesC00_V00
        ---C00_V0000
            ----000000.jpg
            ----000001.jpg
        ---C00_V0001
            ----000000.jpg
            ----000001.jpg
        ---C00_V0002
            ----000000.jpg
            ----000001.jpg
    --KeyFramesC01_V00
        ---C01_V0000
            ----000000.jpg
            ----000001.jpg
        ---C01_V0001
            ----000000.jpg
            ----000001.jpg
        ---C01_V0002
            ----000000.jpg
            ----000001.jpg
    --KeyFramesC02_V00
        ---C02_V0000
            ----000000.jpg
            ----000001.jpg
        ---C02_V0001
            ----000000.jpg
            ----000001.jpg
        ---C02_V0002
            ----000000.jpg
            ----000001.jpg
"""


def create_tables():
    """ create tables in the PostgreSQL database"""

    commands = (
        """
        CREATE TABLE video (
            id BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
            video_id CHAR(15) NOT NULL,
            frame_id CHAR(15) NOT NULL,
            PRIMARY KEY(video_id, frame_id)
        )
        """,
        """ CREATE TABLE ObjectDetection (
                id BIGINT,
                FOREIGN KEY (id) references video(id),
                PRIMARY KEY(id),
                collection TEXT
                )

        """,
        """ CREATE TABLE Classify (
                id BIGINT PRIMARY KEY,
                name_class CHAR(10) NOT NULL
            )
        """,
        """ CREATE TABLE OCR (
                id BIGINT PRIMARY KEY,
                FOREIGN KEY (id) references video(id),
                collection TEXT,
                id_classify INT,
                FOREIGN KEY (id_classify) references classify(id)  )            

        """,
        """ CREATE TABLE ASR (
                id BIGINT PRIMARY KEY,
                FOREIGN KEY (id) references video(id),
                collection TEXT,
                id_classify INT,
                FOREIGN KEY (id_classify) references classify(id)  )          

        """,
        """ CREATE TABLE Place (
                id INT PRIMARY KEY,
                FOREIGN KEY (id) references video(id),
                place TEXT )
        """

    )

    # create table one by one
    for command in commands:
        engine.execute(command)
    # close communication with the PostgreSQL database server
    engine.close()


def insert_infor_video():
    command = """
        INSERT INTO public."video"(video_id, frame_id)
        VALUES('C00_V0000.mp4','000000.jpg'),('C00_V0000.mp4','000004.jpg'),('C00_V0001.mp4','000003.jpg'),('C00_V0001.mp4','000006.jpg'),('C00_V0002.mp4','000000.jpg');
    """
    engine.execute(command)
    engine.close()


if __name__ == '__main__':
    create_tables()
    # insert_infor_video()
