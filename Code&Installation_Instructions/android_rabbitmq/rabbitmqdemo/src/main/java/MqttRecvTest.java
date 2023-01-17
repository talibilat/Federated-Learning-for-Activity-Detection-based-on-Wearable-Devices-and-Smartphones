// Heavily based on RabbitMQ MQTT adapter test case code!

// first, import the RabbitMQ Java client
// and the Paho MQTT client classes, plus any other
// requirements

import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import org.eclipse.paho.client.mqttv3.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.concurrent.TimeoutException;

/***
 *  MQTT v3.1 tests
 *  TODO: synchronise access to variables
 */

public class MqttRecvTest implements MqttCallback {

    // setup some variables which define where the MQTT broker is
    private final String host = "10.0.1.6";
    private final String username = "admin";
    private final String password = "g150";
    private final int port = 1883;
    private final String brokerUrl = "tcp://" + host + ":" + port;
    private String clientId;
    private MqttClient client;
    private MqttConnectOptions conOpt;

    // specify a message payload - doesn't matter what this says, but since MQTT expects a byte array
    // we convert it from string to byte array here
    private final byte[] payload = "this payload was published on MQTT and read using AMQP".getBytes();

    // specify the topic to be used
    private final String topic = "test-topic";

    private int testDelay = 5000;
    private long lastReceipt;
    private boolean expectConnectionFailure;


    // override 10s limit
    private class MyConnOpts extends MqttConnectOptions {
        private int keepAliveInterval = 60;

        @Override
        public void setKeepAliveInterval(int keepAliveInterval) {
            this.keepAliveInterval = keepAliveInterval;
        }

        @Override
        public int getKeepAliveInterval() {
            return keepAliveInterval;
        }
    }

    public void setUpMqtt() throws MqttException {
        clientId = getClass().getSimpleName() + ((int) (10000 * Math.random()));
        client = new MqttClient(brokerUrl, clientId);
        conOpt = new MyConnOpts();
        setConOpts(conOpt);

        expectConnectionFailure = false;
    }

    public void tearDownMqtt() throws MqttException {
        // clean any sticky sessions
        setConOpts(conOpt);
        client = new MqttClient(brokerUrl, clientId);
        try {
            client.connect(conOpt);
            client.disconnect();
        } catch (Exception e) {
        }

    }

    private void setConOpts(MqttConnectOptions conOpts) {
        // provide authentication if the broker needs it
        conOpts.setUserName(username);
        conOpts.setPassword(password.toCharArray());
        conOpts.setCleanSession(true);
        conOpts.setKeepAliveInterval(60);
    }


    public void connectionLost(Throwable cause) {
        if (!expectConnectionFailure)
            System.out.println("Connection unexpectedly lost");
    }

    @Override
    public void messageArrived(String topic, MqttMessage message) throws Exception {
        String time = new Timestamp(System.currentTimeMillis()).toString();
        System.out.println("Time:\t" + time + "  Topic:\t" + topic + "  Message:\t" + new String(message.getPayload()));

        String path = "src/main/resources/saved_file.txt";
        File file = new File(path);
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(message.getPayload());

            //fos.close // no need, try-with-resources auto close
        }
    }

    public void deliveryComplete(IMqttDeliveryToken token) {
    }

    public void run() {
        try {

            setUpMqtt(); // initialise the MQTT connection

            client.setCallback(this);

            client.connect(conOpt);
//
//            publish(client, topic, 1, payload); // publish the MQTT message

            client.subscribe(topic, 1);

            // Continue waiting for messages until the Enter is pressed
            System.out.println("Press <Enter> to exit");
            try
            {
                System.out.println("Started recieving");
                 System.in.read();
            }
            catch (IOException e)
            {
                // If we can't read we'll just exit
            }
            Thread.sleep(testDelay);

            client.disconnect();
//
            tearDownMqtt(); // cleanup MQTT resources

        } catch (Exception mqe) {
            mqe.printStackTrace();
        }
    }


    public static void main(String[] args) {
        MqttRecvTest mqt = new MqttRecvTest();
        mqt.run();
    }


}
