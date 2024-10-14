#include <sys/stat.h>
#include <stdbool.h>
//#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <getopt.h>
#include "header.h"

static int help_flag;
static int repetitionsB = 1;
static bool performance = false;
static int versionV = 0;
static int length = 0;
const char *helpMessage =
    "Verwendung: [-h | --help] | [-V version] [-B repetitions]\n"
    " -h:     Ausgabe dieser Hilfnachricht und eines Verwendungsbeispiels\n"
    " -help:  Ausgabe dieser Hilfnachricht und eines Verwendungsbeispiels\n"
    " -V:     version (0 - 7) Bestimmt die verwendete Implementierung\n"
    " -B:     repetitions (1 - 2147483647) Die Laufzeit wird gemessen, das übergebene optionale Argument gibt dabei die Anzahl an Wiederholungen an\n"
    "   Versionen:\n"
    "       -V0 - LUT (uneven, polynomial)\n"
    "       -V1 - naive\n"
    "       -V2 - Series\n"
    "       -V3 - LUT-SIMD (uneven, polynomial)\n"
    "       -V4 - Series-SIMD\n"
    "       -V5 - LUT (uneven, linear)\n"
    "       -V6 - LUT (even, polynomial)\n"
    "       -V7 - LUT (even, linear)\n";
const char *usageExample =
    "Verwendungsbeispiel:\n"
    "             [./main] -V1 -B3\n"
    " -V1   --> erste Vergleichsimplementierung (2.Implementierung) wird genutzt\n"
    " -B    --> Laufzeit wird gemessen und ausgegeben\n"
    " -B3   --> Funktionsaufruf wird 3 mal wiederholt.\n";
const char *errorMessage =
    "Falsche Argumente eingegeben! Nutzen sie \"[./main] --help\" für weitere Informationen\n";

void printHelpMessage()
{
    printf("%s\n%s\n", helpMessage, usageExample);
    help_flag = 0;
}

void printError()
{
    fprintf(stderr, "%s\n", errorMessage);
}

// wandelt einen char-Array in ein float-Array um
float *stringToFloats(char *ptr)
{
    float *arr;
    if ((arr = malloc(200)) == NULL)
    {
        fprintf(stderr, "Fehler beim Umwandeln der Eingabedatei in Floats (malloc)\n");
        exit(EXIT_FAILURE);
    }
    bool readFloat = false;
    int count = 0;

    if (ptr[0] != '\n' && ptr[0] != '\0' && ptr[0] != ' ')
    {
        arr[count] = (float)atof(ptr);
        ptr++;
        count++;
    }

    // floats werden eingelesen solange das Nullbyte noch nicht erreicht worden ist
    while (ptr[0] != '\0')
    {
        if (readFloat)
        {
            arr[count] = (float)atof(ptr);
            count++;
            readFloat = false;

            // falls das Array zu klein ist wird es vergrößert
            if (count % 50 == 0)
            {
                float *help;
                if ((help = malloc((count * 4) + 200)) == NULL)
                {
                    return NULL;
                }
                float *delete = arr;
                memcpy(help, arr, count * 4);
                arr = help;
                free(delete);
            }
        }
        // wenn ein Leerzeichen oder eine neue Zeile gelesen wird, dann folgt danach ein float-Wert, weshalb readFloat auf true gesetzt wird
        if (ptr[0] == '\n' || ptr[0] == ' ')
        {
            readFloat = true;
        }
        ptr++;
    }
    length = count;
    // float-Werte werden in Array mit passender Größe geschrieben
    float *help = malloc(count * 4);
    float *delete = arr;
    memcpy(help, arr, count * 4);
    arr = help;
    free(delete);
    for (int i = 0; i < count; i++)
    {
        if (arr[i] < 0 || arr[i] == -(1.0/0.0))
        {
            fprintf(stderr, "Fehler! Eingabewerte dürfen nicht kleiner als 0 sein!\n");
            exit(EXIT_FAILURE);
        }
        if(arr[i] > 1 || arr[i] == (1.0/0.0)){
            fprintf(stderr, "Fehler! Eingabewerte dürfen nicht größer als 1 sein!\n");
            exit(EXIT_FAILURE);     
        }
        if(arr[i] != arr[i]){
            fprintf(stderr, "Fehler! Eingabewert ist \"Not a Number!\". Eingabewerte überprüfen!\n");
            exit(EXIT_FAILURE);
        }
    }
    return arr;
}

void getOptions(int argc, char **argv)
{
    char flag;
    static struct option long_options[] = {
        {"help", no_argument, &help_flag, 1},
        {"Version", required_argument, 0, 'V'},
        {"Benchmark", required_argument, 0, 'B'},
        {"h", no_argument, 0, 'h'},
        {0, 0, 0, 0} // notwendig???
    };
    int optionIndex = 0;
    bool wrong_Flag = false;
    // Einlesen der Flags und der Argumente
    while ((flag = getopt_long(argc, argv, "V:B::h", long_options, &optionIndex)) != -1)
    {
        switch (flag)
        {
        // Helpmessage wird ausgegeben
        case 0:
            // eigentlich ist hier --help schon eingetreten
            if (*long_options[optionIndex].flag == 1)
            {
                printHelpMessage();
                exit(EXIT_SUCCESS);
            }
            break;
        // Helpmessage wird ausgegeben
        case 'h':
            printHelpMessage();
            exit(EXIT_SUCCESS);
            break;
        // Version wird bestimmt
        case 'V':
            versionV = (int)atol(optarg);
            if (versionV < 0 || versionV > 7)
            {
                fprintf(stderr, "Ungültige Version. \"Version %d\" existiert nicht\n"
                                "Zur korrekten Nutzung \"./main -h\" aufrufen\n",
                        versionV);
                exit(EXIT_FAILURE);
            }
            break;
        // Performancemessung und Anzahl an Wiederholungen wird bestimmt
        case 'B':
            performance = true;
            if (optarg == 0)
            {
                break;
            }
            if (0 == (int)atol(optarg))
            {
                fprintf(stderr, "Ungültige Eingabe! -B<int> sollte zwischen 1 und 2147483647 liegen!\n");
                exit(EXIT_FAILURE);
                break;
            }
            repetitionsB = (int)atol(optarg);
            if (repetitionsB < 0)
            {
                fprintf(stderr, "Ungültige Eingabe! -B<int> sollte zwischen 1 und 2147483647 liegen!\n");
                exit(EXIT_FAILURE);
            }
            break;
        case '?':
            wrong_Flag = true;
            break;
        default:
            break;
        }
    }
    //bool-Variable wird genutzt, damit falls -h oder --help gesetzt ist die Fehldermeldung ausgegeben wird und kein Fehler geworfen wird
    if (wrong_Flag)
    {
        printError();
        exit(EXIT_FAILURE);
    }
    return;
}

// Berechnet die Entropy mit der übergebenen Funktion; falls notwendig wird die Zeit gemessen
float calculate(const float data[], float (*entropy)(size_t, const float[]))
{
    // Zeit wird nicht gemessen --> Entropie wird zurückgegeben
    if (!performance)
    {
        return (*entropy)(length, data);
    }

    float entropy_Value, time;
    struct timespec start, end;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0)
    {
        fprintf(stderr, "Fehler bei der Zeitmessung \"clock_gettime\"\n");
        exit(EXIT_FAILURE);
    }

    // Entropieberechnung wird mehrfach ausgeführt
    for (int i = 0; i < repetitionsB; i++)
    {
        entropy_Value = (*entropy)(length, data);
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0)
    {
        fprintf(stderr, "Fehler bei der Zeitmessung \"clock_gettime\"\n");
        exit(EXIT_FAILURE);
    }
    // Berechnung und Ausgabe der Zeit
    time = end.tv_sec - start.tv_sec + (1e-9) * (end.tv_nsec - start.tv_nsec);
    printf("Dauer von V%d: %f\n", versionV, time);
    // Rückgabe der Entropie
    return entropy_Value;
}

// bestimmt die verwendete entropy-Version und gibt das Ergebnis aus
void start(const float data[])
{
    if(length == 0){
        fprintf(stderr, "Fehler bei der Entropieberechnung! Die Arraylänge ist kleiner als 1\n");
        exit(EXIT_FAILURE);    
    }
    float result;
    switch (versionV)
    {
    case 0: // LUT
        result = calculate(data, entropy);
        break;
    case 1: // naive implementation
        result = calculate(data, entropy_V1);
        break;
    case 2: // series implementation
        result = calculate(data, entropy_V2);
        break;
    case 3: // LUT implementation with SIMD
        result = calculate(data, entropy_V3);
        break;
    case 4: // series implementation with SIMD
        result = calculate(data, entropy_V4);
        break;
    case 5: // LUT uneven linear
        result = calculate(data, entropy_V5);
        break;
    case 6: // LUT even polynomial
        result = calculate(data, entropy_V6);
        break;
    case 7: // LUT even linear
        result = calculate(data, entropy_V7);
        break;
    // sollte eigentlich nie auftreten
    default:
        fprintf(stderr, "Fehler bei der Versionswahl!\n");
        exit(EXIT_FAILURE);
    }
    printf("Die berechnete Entropie beträgt: %f\n", result);
}

int main(int argc, char **argv)
{
    char help;
    char *path;
    FILE *readFile;
    char *data;
    size_t dataSize;
    float *floatArray;
    struct stat fileInformation;
    bool data_found = false;
    // Iteration durch argv um die Eingabedatei zu finden
    for (int i = 1; i <= argc; i++)
    {
        path = argv[i];
        if (path == NULL)
        { // notwendig evtl Schleife anpassen um null-case zu umgehen oder Fehlermeldung?
            break;
        }
        help = path[0];

        // wenn erstes char kein '-' ist, dann ist es die Eingabedatei
        if (help != '-')
        {
            data_found = true;
            // Dateigröße ermitteln und Inhalt in ein char-Array kopieren
            if ((readFile = fopen(path, "r")) == NULL)
            {
                fprintf(stderr, "Kein Lesen aus der Datei %s möglich, bitte eine gültige Datei angeben\n", path);
                exit(EXIT_FAILURE);
            }
            if (stat(path, &fileInformation) != 0)
            {
                fprintf(stderr, "Fehler bei fstat\n");
                exit(EXIT_FAILURE);
            }

            dataSize = (size_t)fileInformation.st_size;

            if ((data = malloc(dataSize + 1)) == NULL)
            {
                fprintf(stderr, "Fehler beim Einlesen der Datei(malloc)\n");
                exit(EXIT_FAILURE);
            }
            fread(data, 1, dataSize + 1, readFile);
            fclose(readFile);
            if ((floatArray = stringToFloats(data)) == NULL)
            {
                fprintf(stderr, "Fehler beim Umwandeln der Datei in Floats\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    // Ermitteln der gesetzten Flags und deren übergebenen Argumente
    getOptions(argc, argv);
    if (!data_found)
    {
        fprintf(stderr, "Keine Eingabedatei vorhanden, \"./main -h\" für Hilfe nutzen\n");
        exit(EXIT_FAILURE);
    }
    // start der Berechnung
    start(floatArray);

    return EXIT_SUCCESS;
}