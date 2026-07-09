from db.connection import connect_oracle, disconnect_oracle
from ui.menu import main_menu

def main():
    conn = connect_oracle()
    if not conn:
        return
    cursor = conn.cursor()
    main_menu(conn, cursor)
    cursor.close()
    disconnect_oracle(conn)

if __name__ == "__main__":
    main()