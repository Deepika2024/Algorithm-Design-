package JavaSocketChat;
import java.io.*;
import java.net.*;

public class Client {

    public static void main(String[] args) {

        String host = "192.168.128.42";
        int port = 5000;

        try {

            System.out.println("Connecting to server...");

            Socket socket = new Socket(host, port);

            System.out.println("Connected to server!");

            BufferedReader input = new BufferedReader(
                    new InputStreamReader(socket.getInputStream()));

            PrintWriter output = new PrintWriter(
                    socket.getOutputStream(), true);

            output.println("Hello ");

            String response = input.readLine();

            System.out.println("Server: " + response);

            socket.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}