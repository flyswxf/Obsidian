import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// --- Main Demo Class ---

public class PersonImplementation {
    public static void main(String[] args) {
        System.out.println("=== Simulation Start ===");

        scenario1_WeatherStorm();
        scenario2_NaughtyElf();
        scenario3_DuckEcology();
        scenario4_MutantResurrection();
        scenario5_AircraftInteraction();
        scenario6_IndoorShelter();
        scenario7_ExtremeFlight();
    }

    // 1. Weather Change (Storm)
    public static void scenario1_WeatherStorm() {
        System.out.println("\n=== Scenario 1: Weather Change (Storm) ===");

        Weather weather = new Weather(); // Default Sunny
        DuckFactory duckFactory = new DuckFactory();

        EnvironmentAdaptiveSuit boySuit = new EnvironmentAdaptiveSuit(new Boy());
        boySuit.getPerson().display();

        Duck duck = duckFactory.createDuck("MALLARD");

        // Register Observers to Weather
        weather.registerObserver(boySuit);
        weather.registerObserver(duck);

        System.out.println("Weather forecast: Storm coming!");
        weather.setWeather(WeatherType.STORMY);
    }

    // 2. Naughty Elf
    public static void scenario2_NaughtyElf() {
        System.out.println("\n=== Scenario 2: Naughty Elf ===");
        Weather weather = new Weather();

        EnvironmentAdaptiveSuit girlSuit = new EnvironmentAdaptiveSuit(new Girl());
        weather.registerObserver(girlSuit);
        girlSuit.getPerson().display();

        // Initial state
        weather.setWeather(WeatherType.RAIN);

        NaughtyElf elf = new NaughtyElf(weather);
        elf.display();

        System.out.println("Naughty Elf casts a spell on " + girlSuit.getPerson().getName() + "!");
        Magic magic = elf.disableAutoSwitch(girlSuit);
        weather = elf.changeWeather(WeatherType.SUNNY);

        System.out.println("20s passed");
        magic.notifyObservers();
        weather.notifyObservers();
    }

    // 3. Duck Ecology
    public static void scenario3_DuckEcology() {
        System.out.println("\n=== Scenario 3: Duck Ecology ===");
        Weather weather = new Weather();

        DuckFactory duckFactory = new DuckFactory();
        Duck mallard = duckFactory.createDuck("MALLARD");
        weather.registerObserver(mallard);

        weather.setWeather(WeatherType.SUNNY); // Duck comes out

        EnvironmentAdaptiveSuit hunterSuit = new EnvironmentAdaptiveSuit(new Hunter());
        weather.registerObserver(hunterSuit);
        hunterSuit.getPerson().display();
        ((Hunter) hunterSuit.getPerson()).shoot(mallard);

        System.out.println("System triggering resurrection...");
        mallard = mallard.resurrect();
        mallard.display();
    }

    // 4. Mutant Resurrection
    public static void scenario4_MutantResurrection() {
        System.out.println("\n=== Scenario 4: Mutant Resurrection ===");
        // Try multiple times to see mutation
        for (int i = 1; i <= 3; i++) {
            System.out.println("\nAttempt " + i + ":");
            Duck d = new MallardDuck();
            d.shot();
            d = d.resurrect();
            d.display();
            d.performFly();
            d.performQuack();
        }
    }

    // 5. Aircraft Interaction
    public static void scenario5_AircraftInteraction() {
        System.out.println("\n=== Scenario 5: Aircraft Interaction ===");
        BoeingFactory boeingFactory = new BoeingFactory();
        Aircraft boeing = boeingFactory.createAircraft();
        ApacheFactory apacheFactory = new ApacheFactory();
        Aircraft apache = apacheFactory.createAircraft();
        EnemyAircraftFactory enemyAircraftFactory = new EnemyAircraftFactory();
        Aircraft enemy = enemyAircraftFactory.createAircraft();

        boeing.display();
        apache.display();
        boeing.performFly();
        apache.performFly();

        enemy.display();
        enemy.performFly();
        ((Apache) apache).attack(enemy);
    }

    // 6. Indoor Shelter
    public static void scenario6_IndoorShelter() {
        System.out.println("\n=== Scenario 6: Indoor Shelter ===");
        EnvironmentAdaptiveSuit boySuit = new EnvironmentAdaptiveSuit(new Boy());
        EnvironmentAdaptiveSuit girlSuit = new EnvironmentAdaptiveSuit(new Girl());
        boySuit.getPerson().display();
        girlSuit.getPerson().display();

        BoeingFactory boeingFactory = new BoeingFactory();
        Aircraft boeing = boeingFactory.createAircraft();
        boeing.registerObserver(boySuit.getPerson());
        boeing.registerObserver(girlSuit.getPerson());
        boeing.display();
        boeing.performFly();
    }

    // 7. Extreme Flight
    public static void scenario7_ExtremeFlight() {
        System.out.println("\n=== Scenario 7: Extreme Flight ===");
        Weather weather = new Weather();

        EnvironmentAdaptiveSuit boySuit = new EnvironmentAdaptiveSuit(new Boy());
        EnvironmentAdaptiveSuit girlSuit = new EnvironmentAdaptiveSuit(new Girl());
        boySuit.getPerson().display();
        girlSuit.getPerson().display();

        weather.registerObserver(boySuit);
        weather.registerObserver(girlSuit);

        System.out.println("Found Personal Flying Wings in the house!");
        boySuit.putOn(PersonalFlyingWing.class);
        boySuit.setAsBase(); // Lock wings as base

        girlSuit.putOn(PersonalFlyingWing.class);
        girlSuit.setAsBase();

        boySuit.getPerson().performFly();
        girlSuit.getPerson().performFly();

        weather.setWeather(WeatherType.HIGH_ALTITUDE); // Should add oxygen mask
    }
}

// --- Interfaces & Enums ---

interface FlyBehavior {
    void fly();
}

interface Shootable {
    void shot();
}

enum WeatherType {
    HOT, COLD, RAIN, STORMY, SUNNY, FIRE, HIGH_ALTITUDE
}

interface WeatherObserver {
    void onWeatherChange(WeatherType type);
}

// --- Weather Subject ---

class Weather {
    private WeatherType currentWeather;
    private List<WeatherObserver> observers = new ArrayList<>();

    public void registerObserver(WeatherObserver o) {
        observers.add(o);
    }

    public void removeObserver(WeatherObserver o) {
        observers.remove(o);
    }

    public void setWeather(WeatherType type) {
        System.out.println("Setting weather to: " + type);
        this.currentWeather = type;
        notifyObservers();
    }

    public void notifyObservers() {
        for (WeatherObserver observer : observers) {
            observer.onWeatherChange(currentWeather);
        }
    }
}

// --- Strategies ---

class FlyNoWay implements FlyBehavior {
    public void fly() {
        System.out.println("I can't fly.");
    }
}

class FlyWithWings implements FlyBehavior {
    public void fly() {
        System.out.println("I'm flying with wings!");
    }
}

class FlyWithPropulsion implements FlyBehavior {
    public void fly() {
        System.out.println("I'm flying with propulsion (Zoom)!");
    }
}

class FlyRocketPowered implements FlyBehavior {
    public void fly() {
        System.out.println("I'm flying with rocket power!");
    }
}

class FlyGhost implements FlyBehavior {
    public void fly() {
        System.out.println("I'm floating through walls!");
    }
}

class FlyWithRotors implements FlyBehavior {
    public void fly() {
        System.out.println("I'm flying with rotors!");
    }
}

class QuackBehavior {
    public void quack() {
        System.out.println("Quack");
    }
}

class Quack extends QuackBehavior {
    public void quack() {
        System.out.println("Quack");
    }
}

class Squick extends QuackBehavior {
    public void quack() {
        System.out.println("Squick");
    }
}

class MuteQuack extends QuackBehavior {
    public void quack() {
        System.out.println("<< Silence >>");
    }
}

class QuackRobotic extends QuackBehavior {
    public void quack() {
        System.out.println("Beep Boop Quack");
    }
}

// --- Aircraft ---
interface AircraftObserver {
    void onAircraftSpotted(Aircraft aircraft);
}

abstract class Aircraft implements Shootable {
    private List<AircraftObserver> observers = new ArrayList<>();
    protected FlyBehavior flyBehavior;

    protected boolean isCivil;
    protected String name;

    public void shot() {
        if (isCivil) {
            System.out.println("[System] WARNING: Civil aircraft " + name + " cannot be shot down! Attack blocked.");
        } else {
            System.out.println(name + " shot down and falling!");
        }
    }

    public void performFly() {
        flyBehavior.fly();
        notifyObservers();
    }

    public void setFlyBehavior(FlyBehavior fb) {
        this.flyBehavior = fb;
    }

    public void registerObserver(AircraftObserver o) {
        observers.add(o);
    }

    public void removeObserver(AircraftObserver o) {
        observers.remove(o);
    }

    protected void notifyObservers() {
        for (AircraftObserver observer : observers) {
            observer.onAircraftSpotted(this);
        }
    }

    public void display() {
        System.out.println("I am a " + name);
    }

}

class Boeing extends Aircraft {
    public Boeing() {
        this.name = "Boeing 747";
        this.isCivil = true;
        this.flyBehavior = new FlyWithPropulsion();
    }

    public void display() {
        System.out.println("I am a Boeing 747 (Civil).");
    }
}

class Apache extends Aircraft {
    public Apache() {
        this.name = "Apache Helicopter";
        this.isCivil = false;
        this.flyBehavior = new FlyWithRotors();
    }

    public void display() {
        System.out.println("I am an Apache Helicopter (Military).");
    }

    public void attack(Shootable target) {
        System.out.println("Apache fires missile!");
        target.shot();
    }
}

class EnemyAircraft extends Aircraft {
    public EnemyAircraft() {
        this.name = "Enemy Fighter";
        this.isCivil = false;
        this.flyBehavior = new FlyWithPropulsion();
    }

    public void display() {
        System.out.println("I am an Enemy Fighter.");
    }
}

interface AircraftFactory {
    Aircraft createAircraft();
}

class BoeingFactory implements AircraftFactory {
    @Override
    public Aircraft createAircraft() {
        return new Boeing();
    }
}

class ApacheFactory implements AircraftFactory {
    @Override
    public Aircraft createAircraft() {
        return new Apache();
    }
}

class EnemyAircraftFactory implements AircraftFactory {
    @Override
    public Aircraft createAircraft() {
        return new EnemyAircraft();
    }
}

// --- Duck ---

class Duck implements Shootable, WeatherObserver {
    protected FlyBehavior flyBehavior;
    protected QuackBehavior quackBehavior;
    protected String name = "Duck";
    protected boolean isAlive = true;

    public Duck() {
        flyBehavior = new FlyWithWings();
        quackBehavior = new Quack(); // Default Quack
    }

    public void display() {
        System.out.println("I am a " + name);
    }

    public void performFly() {
        flyBehavior.fly();
    }

    public void performQuack() {
        quackBehavior.quack();
    }

    public void shot() {
        if (isAlive) {
            System.out.println(name + " was shot and died.");
            isAlive = false;
        } else {
            System.out.println(name + " is already dead.");
        }
    }

    public Duck resurrect() {
        if (isAlive) {
            System.out.println(name + " is already alive.");
            return this;
        }

        System.out.println(name + " resurrects!");

        // Mutation Logic
        Random rand = new Random();
        int chance = rand.nextInt(100); // 0-99
        Duck newDuck;
        DuckFactory duckFactory = new DuckFactory();
        if (chance < 20) { // 20% chance for Ghost
            System.out.println("It mutated into a Ghost Duck!");
            newDuck = duckFactory.createDuck("Ghost");
        } else if (chance < 40) { // 20% chance for Mechanical (20-39)
            System.out.println("It mutated into a Mechanical Duck!");
            newDuck = duckFactory.createDuck("Mechanical");
        } else {
            System.out.println("It came back as a normal duck.");
            newDuck = duckFactory.createDuck("Mallard");
        }
        return newDuck;
    }

    @Override
    public void onWeatherChange(WeatherType type) {
        if (!isAlive)
            return;

        if (type == WeatherType.STORMY || type == WeatherType.RAIN) {
            System.out.println(name + " is seeking shelter from the bad weather!");
        } else if (type == WeatherType.SUNNY) {
            System.out.println(name + " comes out to play in the sun.");
        }
    }
}

class MallardDuck extends Duck {
    public MallardDuck() {
        this.name = "Mallard Duck";
        this.quackBehavior = new Quack();
        this.flyBehavior = new FlyWithWings();
    }
}

class RubberDuck extends Duck {
    public RubberDuck() {
        this.name = "Rubber Duck";
        this.quackBehavior = new Squick();
        this.flyBehavior = new FlyNoWay();
    }
}

class GhostDuck extends Duck {
    public GhostDuck() {
        this.name = "Ghost Duck";
        this.quackBehavior = new MuteQuack();
        this.flyBehavior = new FlyGhost();
    }
}

class MechanicalDuck extends Duck {
    public MechanicalDuck() {
        this.name = "Mechanical Duck";
        this.quackBehavior = new QuackRobotic();
        this.flyBehavior = new FlyRocketPowered();
    }
}

class RedHeadDuck extends Duck {
    public RedHeadDuck() {
        this.name = "Red Head Duck";
        this.quackBehavior = new Quack();
        this.flyBehavior = new FlyWithWings();
    }
}

class DecoyDuck extends Duck {
    public DecoyDuck() {
        this.name = "Decoy Duck";
        this.quackBehavior = new MuteQuack();
        this.flyBehavior = new FlyNoWay();
    }
}

class DuckFactory {
    public Duck createDuck(String type) {
        switch (type) {
            case "MALLARD":
                return new MallardDuck();
            case "RUBBER":
                return new RubberDuck();
            case "GHOST":
                return new GhostDuck();
            case "MECHANICAL":
                return new MechanicalDuck();
            case "REDHEAD":
                return new RedHeadDuck();
            case "DECOY":
                return new DecoyDuck();
            default:
                return new Duck();
        }
    }
}

// --- Magic & Elf ---

interface MagicObserver {
    void onMagicEventReleased();
}

class Magic {
    private List<MagicObserver> observers = new ArrayList<>();

    public void addObserver(MagicObserver o) {
        observers.add(o);
    }

    public void notifyObservers() {
        for (MagicObserver o : observers) {
            o.onMagicEventReleased();
        }
    }
}

class NaughtyElf {
    private Weather weather;

    public NaughtyElf(Weather weather) {
        this.weather = weather;
    }

    public Magic disableAutoSwitch(EnvironmentAdaptiveSuit suit) {
        Magic magic = new Magic();
        suit.disableAutoSwitch();
        magic.addObserver(suit);
        return magic;
    }

    public Weather changeWeather(WeatherType type) {
        System.out.println("Naughty Elf changes the weather!");
        weather.setWeather(type);
        return weather;
    }

    public void display() {
        System.out.println("I am a Naughty Elf");
    }
}

// --- Component: Person ---

abstract class Person implements AircraftObserver {
    protected String name;

    public String getName() {
        return name;
    }

    public void display() {
        System.out.println("I am " + name);
    }

    public void performFly() {
        System.out.println("I can't fly.");
    }

    @Override
    public void onAircraftSpotted(Aircraft aircraft) {
        System.out.println(name + " spots an aircraft!");
        System.out.println(name + " runs and hides inside the house.");
    }
}

class Boy extends Person {
    public Boy() {
        this.name = "Boy";
    }
}

class Girl extends Person {
    public Girl() {
        this.name = "Girl";
    }
}

class Hunter extends Person {
    public Hunter() {
        this.name = "Hunter";
    }

    public void shoot(Shootable target) {
        System.out.println("Hunter shoots!");
        target.shot();
    }
}

// --- Decorator ---

abstract class ClothingDecorator extends Person {
    protected Person person;

    public ClothingDecorator(Person person) {
        this.person = person;
    }

    @Override
    public String getName() {
        return person.getName();
    }

    @Override
    public void display() {
        person.display();
    }

    @Override
    public void performFly() {
        person.performFly();
    }
}

// --- Concrete Decorators ---

class RainCoat extends ClothingDecorator {
    public RainCoat(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Rain Coat");
    }
}

class WaterproofShoes extends ClothingDecorator {
    public WaterproofShoes(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Waterproof Shoes");
    }
}

class Shirt extends ClothingDecorator {
    public Shirt(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Shirt");
    }
}

class Jeans extends ClothingDecorator {
    public Jeans(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Jeans");
    }
}

class Sneaker extends ClothingDecorator {
    public Sneaker(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Sneaker");
    }
}

class OxygenMask extends ClothingDecorator {
    public OxygenMask(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Oxygen Mask");
    }
}

class CasualSuit extends ClothingDecorator {
    public CasualSuit(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Casual Suit");
    }
}

class DownJacket extends ClothingDecorator {
    public DownJacket(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Down Jacket");
    }
}

class Boots extends ClothingDecorator {
    public Boots(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Boots");
    }
}

class FireproofSuit extends ClothingDecorator {
    public FireproofSuit(Person p) {
        super(p);
    }

    public void display() {
        super.display();
        System.out.println(" + Wearing Fireproof Suit");
    }
}

class PersonalFlyingWing extends ClothingDecorator {
    private FlyBehavior myFlyBehavior = new FlyWithWings();

    public PersonalFlyingWing(Person p) {
        super(p);
    }

    public void performFly() {
        myFlyBehavior.fly();
    }

    public void display() {
        super.display();
        System.out.println(" + Equipped with Personal Flying Wing");
    }
}

class FlyingMotorcycle extends ClothingDecorator {
    private FlyBehavior myFlyBehavior = new FlyWithPropulsion();

    public FlyingMotorcycle(Person p) {
        super(p);
    }

    public void performFly() {
        myFlyBehavior.fly();
    }

    public void display() {
        super.display();
        System.out.println(" + Riding a Flying Motorcycle");
    }
}

// --- Environment Adaptive Suit ---

class EnvironmentAdaptiveSuit implements MagicObserver, WeatherObserver {
    private Person basePerson;
    private Person currentPerson;
    private boolean autoSwitchEnabled = true;

    public EnvironmentAdaptiveSuit(Person base) {
        this.basePerson = base;
        this.currentPerson = base;
    }

    public Person getPerson() {
        return currentPerson;
    }

    public void setAsBase() {
        this.basePerson = this.currentPerson;
    }

    public void resetClothes() {
        if (!autoSwitchEnabled) {
            return;
        }
        this.currentPerson = basePerson;
    }

    public void putOn(Class<? extends ClothingDecorator> clothingClass) {
        if (!autoSwitchEnabled) {
            System.out.println("[Suit] Auto-switch is disabled. Cannot change clothes.");
            return;
        }
        try {
            this.currentPerson = clothingClass.getConstructor(Person.class).newInstance(this.currentPerson);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void disableAutoSwitch() {
        this.autoSwitchEnabled = false;
    }

    public void enableAutoSwitch() {
        this.autoSwitchEnabled = true;
    }

    @Override
    public void onMagicEventReleased() {
        System.out.println("Magic Released");
        enableAutoSwitch();
    }

    @Override
    public void onWeatherChange(WeatherType type) {
        this.resetClothes();

        switch (type) {
            case RAIN:
            case STORMY:
                this.putOn(RainCoat.class);
                this.putOn(WaterproofShoes.class);
                break;
            case SUNNY:
            case HOT:
                this.putOn(Shirt.class);
                this.putOn(Sneaker.class);
                break;
            case HIGH_ALTITUDE:
                this.putOn(OxygenMask.class);
                break;
            case FIRE:
                this.putOn(FireproofSuit.class);
                break;
            default:
                this.putOn(CasualSuit.class);
                break;
        }
        this.getPerson().display();
    }
}
