package utils;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
 
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 *
 * @author DanElias
 */
public class JSONUtils {
    
    public static void printJsonObject(JSONObject jsonObject, String key) {
        // A JSON array. JSONObject supports java.util.List interface.
        JSONArray companyList = (JSONArray) jsonObject.get(key);

        // An iterator over a collection. Iterator takes the place of Enumeration in the Java Collections Framework.
        // Iterators differ from enumerations in two ways:
        // 1. Iterators allow the caller to remove elements from the underlying collection during the iteration with well-defined semantics.
        // 2. Method names have been improved.
        Iterator<JSONObject> iterator = companyList.iterator();
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
    }
    
    public static JSONObject readJson(String json_url) {
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(json_url));

            // A JSON object. Key value pairs are unordered. JSONObject supports java.util.Map interface.
            JSONObject jsonObject = (JSONObject) obj;
            return jsonObject;
        } catch (Exception e) {
            JSONObject jsonObject = new JSONObject();
            System.out.println("We couldn't read the json file");
            e.printStackTrace();
            return jsonObject;
        }
    }
    
    public static List<JSONObject> objectToJSONObjectArrayList(Object obj) {
        List<JSONObject> list;
        if (obj.getClass().isArray()) {
            list = Arrays.asList((JSONObject[])obj);
        } else if (obj instanceof Collection) {
            list = new ArrayList<>((Collection<JSONObject>)obj);
        } else {
            list = Arrays.asList((JSONObject[])obj);
        }
        return list;
    }
    
}
