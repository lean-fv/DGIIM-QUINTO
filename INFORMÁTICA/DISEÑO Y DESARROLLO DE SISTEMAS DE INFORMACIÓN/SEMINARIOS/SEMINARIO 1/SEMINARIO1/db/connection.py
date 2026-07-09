import oracledb

def connect_oracle():
    """
    Establece conexión con Oracle ETSIIT.
    """
    dsn = oracledb.makedsn("oracle0.ugr.es", 1521, service_name="practbd")
    user = input("Usuario (xDNI): ")
    password = input("Contraseña: ")
    try:
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
        conn.autocommit = False
        print("Conexión establecida correctamente.")
        return conn
    except oracledb.Error as e:
        print("Error al conectar:", e)
        return None

def disconnect_oracle(connection):
    """
    Cierra la conexión con Oracle ETSIIT.
    """
    if connection:
        connection.close()
        print("Conexión cerrada.")