#include <M5Unified.h>
#include <WiFi.h>
#include <WiFiClientSecure.h> 
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "modele_chute.h"

// ==========================================
// 1. CONFIGURATION RÉSEAU & MQTT
// ==========================================
const char* ssid      = "";  // Mets ton identifian wifi
const char* password  = "";  // Mettez votre code wifi
const char* mqtt_host = "88ffea9693414c70bc1633c79ad57a2d.s1.eu.hivemq.cloud";
const char* mqtt_user = "";       //  Mets ton vrai User HiveMQ
const char* mqtt_pass = ""; //  Mets ton vrai Mdp HiveMQ)
const int   mqtt_port = 8883;

const char* topic_telemetry = "senior/monitoring/telemetry";
const char* topic_alerte    = "senior/monitoring/alerte";
const char* topic_feedback  = "senior/monitoring/feedback";

WiFiClientSecure espClient; // WiFiClientSecure au lieu de WiFiClient
PubSubClient client(espClient);

// ==========================================
// 2. PARAMÈTRES ML
// ==========================================
#define NUMBER_OF_INPUTS  400
#define NUMBER_OF_OUTPUTS 4
#define TENSOR_ARENA_SIZE (32 * 1024)

const float scaler_mean[4]  = {-0.000361, 0.718680, 0.276890, 1.012660};
const float scaler_scale[4] = { 0.095331, 0.279200, 0.277345, 0.070183};
const String classes[4]     = {"allonge", "assis", "chute", "marche"};

namespace {
  const tflite::Model* tfl_model       = nullptr;
  tflite::MicroInterpreter* tfl_interpreter = nullptr;
  TfLiteTensor* input_tensor    = nullptr;
  TfLiteTensor* output_tensor   = nullptr;
  tflite::MicroMutableOpResolver<6> resolver;
  uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

const int WINDOW_SIZE  = 100;
const int NUM_FEATURES = 4;
float raw_buffer[WINDOW_SIZE][NUM_FEATURES];
float flat_input_array[NUMBER_OF_INPUTS];
float output_array[NUMBER_OF_OUTPUTS];

int    buffer_index       = 0;
String etat_actuel        = "---";
float  confiance_actuelle = 0.0;

// ==========================================
// 3. INTERFACE VISUELLE — CONSTANTES
// ==========================================

// Palette de couleurs
#define COL_BG        0x1082  // Gris très foncé (quasi noir)
#define COL_PANEL     0x2104  // Gris foncé pour les panneaux
#define COL_BORDER    0x4208  // Bordure subtile
#define COL_BLANC     TFT_WHITE
#define COL_VERT      0x07E0  // Vert vif
#define COL_ORANGE    0xFD20  // Orange
#define COL_ROUGE      TFT_RED
#define COL_BLEU      0x051F  // Bleu foncé
#define COL_BLEU_CLAIR 0x07BF // Bleu clair pour accéléro
#define COL_JAUNE     0xFFE0  // Jaune
#define COL_GRIS      0x8410  // Gris texte secondaire

// Dimensions écran Tab5 (1280x800, on utilise 800x480 en landscape)
// Pour M5Stack Tab5, résolution typique en landscape
#define SCR_W  1280
#define SCR_H  720

// Zones de l'interface (layout en 3 colonnes + header)
#define HDR_H      60
#define COL1_X     0
#define COL1_W     320
#define COL2_X     320
#define COL2_W     620
#define COL3_X     940
#define COL3_W     340
#define BODY_Y     (HDR_H + 4)
#define BODY_H     (SCR_H - HDR_H - 4)

// Graphique accéléro
#define GRAPH_X    (COL2_X + 10)
#define GRAPH_Y    (BODY_Y + 50)
#define GRAPH_W    (COL2_W - 20)
#define GRAPH_H    (BODY_H - 100)
#define GRAPH_SAMPLES 200  // Nombre de points affichés

// Historique
#define HIST_MAX   6  // Nombre d'entrées dans l'historique

// ==========================================
// 4. VARIABLES D'INTERFACE
// ==========================================

// Buffer graphique accéléro (valeurs normalisées 0-GRAPH_H)
float graph_x[GRAPH_SAMPLES];
float graph_y[GRAPH_SAMPLES];
float graph_z[GRAPH_SAMPLES];
int   graph_ptr = 0;
bool  graph_full = false;

// Historique des détections
struct HistEntry {
  String etat;
  float  confiance;
  String heure;
};
HistEntry historique[HIST_MAX];
int hist_count = 0;

// Heure simulée (remplace par NTP si réseau dispo)
uint32_t last_time_ms  = 0;
uint8_t  hh = 0, mm = 0, ss = 0;

// Flag redessiner tout (après alerte)
bool redraw_full = true;

// Dernière alerte feedback reçue
String feedback_msg    = "";
bool   show_feedback   = false;
uint32_t feedback_time = 0;

// ==========================================
// 5. FONCTIONS UTILITAIRES UI
// ==========================================

// Retourne la couleur associée à un état
uint32_t couleur_etat(String etat) {
  if (etat == "chute")   return COL_ROUGE;
  if (etat == "marche")  return COL_VERT;
  if (etat == "assis")   return COL_ORANGE;
  if (etat == "allonge") return COL_BLEU_CLAIR;
  return COL_GRIS;
}

// Retourne l'icône (caractère) associée à un état
String icone_etat(String etat) {
  if (etat == "chute")   return "!";
  if (etat == "marche")  return ">";
  if (etat == "assis")   return "=";
  if (etat == "allonge") return "_";
  return "?";
}

// Formate le temps en HH:MM:SS
String format_time() {
  char buf[12];
  snprintf(buf, sizeof(buf), "%02d:%02d:%02d", hh, mm, ss);
  return String(buf);
}

// Met à jour le temps interne (simple compteur)
void update_time() {
  uint32_t now = millis();
  if (now - last_time_ms >= 1000) {
    last_time_ms = now;
    ss++;
    if (ss >= 60) { ss = 0; mm++; }
    if (mm >= 60) { mm = 0; hh++; }
    if (hh >= 24) { hh = 0; }
  }
}

// Dessine un panneau arrondi avec titre
void draw_panel(int x, int y, int w, int h, String titre, uint32_t col_titre) {
  M5.Display.fillRoundRect(x, y, w, h, 8, COL_PANEL);
  M5.Display.drawRoundRect(x, y, w, h, 8, COL_BORDER);
  if (titre.length() > 0) {
    M5.Display.fillRoundRect(x, y, w, 28, 8, col_titre);
    M5.Display.fillRect(x, y + 14, w, 14, col_titre); // carré bas du header arrondi
    M5.Display.setTextColor(COL_BLANC, col_titre);
    M5.Display.setTextSize(1);
    M5.Display.setTextDatum(MC_DATUM);
    M5.Display.drawString(titre, x + w/2, y + 14);
    M5.Display.setTextDatum(TL_DATUM);
  }
}

// Dessine la barre de progression de confiance
void draw_confidence_bar(float conf) {
  int bx = COL1_X + 15;
  int by = BODY_Y + 220;
  int bw = COL1_W - 30;
  int bh = 32;

  // Fond de la barre
  M5.Display.fillRoundRect(bx, by, bw, bh, 6, COL_BORDER);

  // Remplissage proportionnel
  int fill = (int)(conf * bw);
  uint32_t col_bar = (conf > 0.85) ? COL_ROUGE : (conf > 0.6) ? COL_ORANGE : COL_VERT;
  if (fill > 0)
    M5.Display.fillRoundRect(bx, by, fill, bh, 6, col_bar);

  // Valeur en texte
  char buf[10];
  snprintf(buf, sizeof(buf), "%d%%", (int)(conf * 100));
  M5.Display.setTextColor(COL_BLANC, col_bar);
  M5.Display.setTextSize(2);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString(String(buf), bx + bw/2, by + bh/2);
  M5.Display.setTextDatum(TL_DATUM);
}

// ==========================================
// 6. DESSIN INITIAL (structure complète)
// ==========================================
void draw_full_ui() {
  M5.Display.fillScreen(COL_BG);

  // --- HEADER ---
  M5.Display.fillRect(0, 0, SCR_W, HDR_H, COL_PANEL);
  M5.Display.drawFastHLine(0, HDR_H, SCR_W, COL_BORDER);

  // Titre principal
  M5.Display.setTextColor(COL_BLANC, COL_PANEL);
  M5.Display.setTextSize(2);
  M5.Display.setTextDatum(ML_DATUM);
  M5.Display.drawString("  SYSTEME ANTI-CHUTE", 10, HDR_H/2);

  // Séparateurs verticaux entre colonnes
  M5.Display.drawFastVLine(COL2_X - 2, BODY_Y, BODY_H, COL_BORDER);
  M5.Display.drawFastVLine(COL3_X - 2, BODY_Y, BODY_H, COL_BORDER);

  // --- PANNEAU COL1 : État + Confiance ---
  draw_panel(COL1_X + 5, BODY_Y + 5, COL1_W - 10, BODY_H - 10, "ETAT DETECTE", 0x2945);

  // Label état (sera mis à jour)
  M5.Display.setTextColor(COL_GRIS, COL_PANEL);
  M5.Display.setTextSize(1);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString("Attente donnees...", COL1_X + COL1_W/2, BODY_Y + 130);
  M5.Display.setTextDatum(TL_DATUM);

  // Label "CONFIANCE"
  M5.Display.setTextColor(COL_GRIS, COL_PANEL);
  M5.Display.setTextSize(1);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString("CONFIANCE", COL1_X + COL1_W/2, BODY_Y + 200);
  M5.Display.setTextDatum(TL_DATUM);

  // --- PANNEAU COL2 : Graphe accéléro ---
  draw_panel(COL2_X + 5, BODY_Y + 5, COL2_W - 10, BODY_H - 10, "ACCELEROMETRE TEMPS REEL", 0x0228);

  // Axes du graphique
  M5.Display.drawFastHLine(GRAPH_X, GRAPH_Y + GRAPH_H/2, GRAPH_W, COL_BORDER); // axe zéro
  M5.Display.drawFastVLine(GRAPH_X, GRAPH_Y, GRAPH_H, COL_BORDER);

  // Légende
  int leg_y = GRAPH_Y + GRAPH_H + 8;
  M5.Display.fillRect(GRAPH_X, leg_y, 20, 8, TFT_RED);
  M5.Display.setTextColor(COL_BLANC, COL_BG);
  M5.Display.setTextSize(1);
  M5.Display.drawString(" X", GRAPH_X + 22, leg_y);
  M5.Display.fillRect(GRAPH_X + 60, leg_y, 20, 8, COL_VERT);
  M5.Display.drawString(" Y", GRAPH_X + 82, leg_y);
  M5.Display.fillRect(GRAPH_X + 120, leg_y, 20, 8, COL_BLEU_CLAIR);
  M5.Display.drawString(" Z", GRAPH_X + 142, leg_y);

  // --- PANNEAU COL3 : Historique ---
  draw_panel(COL3_X + 5, BODY_Y + 5, COL3_W - 10, BODY_H - 10, "HISTORIQUE", 0x2945);

  M5.Display.setTextDatum(TL_DATUM);
  redraw_full = false;
}

// ==========================================
// 7. MISES À JOUR PARTIELLES (zone par zone)
// ==========================================

// Met à jour la zone état (col 1)
void update_ui_etat() {
  int zx = COL1_X + 5;
  int zy = BODY_Y + 5;
  int zw = COL1_W - 10;

  // Effacer la zone centrale du panneau
  M5.Display.fillRect(zx + 2, zy + 30, zw - 4, 185, COL_PANEL);

  uint32_t col = couleur_etat(etat_actuel);

  // Icône grande
  M5.Display.setTextColor(col, COL_PANEL);
  M5.Display.setTextSize(8);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString(icone_etat(etat_actuel), zx + zw/2, zy + 90);

  // Nom de l'état
  M5.Display.setTextColor(col, COL_PANEL);
  M5.Display.setTextSize(3);
  M5.Display.setTextDatum(MC_DATUM);
  String etat_upper = etat_actuel;
  etat_upper.toUpperCase();
  M5.Display.drawString(etat_upper, zx + zw/2, zy + 160);

  M5.Display.setTextDatum(TL_DATUM);

  // Label "CONFIANCE"
  M5.Display.setTextColor(COL_GRIS, COL_PANEL);
  M5.Display.setTextSize(1);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString("CONFIANCE", zx + zw/2, zy + 200);
  M5.Display.setTextDatum(TL_DATUM);

  // Barre de confiance
  draw_confidence_bar(confiance_actuelle);
}

// Met à jour le header (statuts WiFi/MQTT + heure)
void update_ui_header() {
  // Effacer la zone droite du header
  M5.Display.fillRect(SCR_W - 500, 0, 500, HDR_H, COL_PANEL);

  // Statut WiFi
  bool wifi_ok = (WiFi.status() == WL_CONNECTED);
  M5.Display.setTextSize(1);
  M5.Display.setTextColor(wifi_ok ? COL_VERT : COL_ROUGE, COL_PANEL);
  M5.Display.setTextDatum(MR_DATUM);
  M5.Display.drawString(wifi_ok ? "WiFi OK" : "WiFi --", SCR_W - 290, HDR_H/2 - 10);

  // Statut MQTT
  bool mqtt_ok = client.connected();
  M5.Display.setTextColor(mqtt_ok ? COL_VERT : COL_ROUGE, COL_PANEL);
  M5.Display.drawString(mqtt_ok ? "MQTT OK" : "MQTT --", SCR_W - 290, HDR_H/2 + 10);

  // Heure
  M5.Display.setTextColor(COL_BLANC, COL_PANEL);
  M5.Display.setTextSize(2);
  M5.Display.setTextDatum(MR_DATUM);
  M5.Display.drawString(format_time(), SCR_W - 10, HDR_H/2);

  M5.Display.setTextDatum(TL_DATUM);
}

// Ajoute un point au graphe accéléro et redessine
void update_ui_graph(float ax, float ay, float az) {
  // Stocker dans le buffer circulaire
  float scale = GRAPH_H / 4.0; // +/-2g → GRAPH_H
  graph_x[graph_ptr] = ax;
  graph_y[graph_ptr] = ay;
  graph_z[graph_ptr] = az;
  graph_ptr = (graph_ptr + 1) % GRAPH_SAMPLES;
  if (graph_ptr == 0) graph_full = true;

  int samples = graph_full ? GRAPH_SAMPLES : graph_ptr;
  if (samples < 2) return;

  // Effacer la zone graphique
  M5.Display.fillRect(GRAPH_X + 1, GRAPH_Y, GRAPH_W - 1, GRAPH_H, COL_PANEL);
  M5.Display.drawFastHLine(GRAPH_X, GRAPH_Y + GRAPH_H/2, GRAPH_W, COL_BORDER);

  // Dessiner les 3 courbes
  int step = GRAPH_W / GRAPH_SAMPLES;
  if (step < 1) step = 1;

  for (int i = 1; i < samples; i++) {
    int idx_prev = (graph_full) ? (graph_ptr + i - 1) % GRAPH_SAMPLES : i - 1;
    int idx_curr = (graph_full) ? (graph_ptr + i)     % GRAPH_SAMPLES : i;

    int px = GRAPH_X + (i - 1) * GRAPH_W / samples;
    int cx = GRAPH_X + i       * GRAPH_W / samples;

    int py_x = GRAPH_Y + GRAPH_H/2 - (int)(graph_x[idx_prev] * scale);
    int cy_x = GRAPH_Y + GRAPH_H/2 - (int)(graph_x[idx_curr] * scale);
    int py_y = GRAPH_Y + GRAPH_H/2 - (int)(graph_y[idx_prev] * scale);
    int cy_y = GRAPH_Y + GRAPH_H/2 - (int)(graph_y[idx_curr] * scale);
    int py_z = GRAPH_Y + GRAPH_H/2 - (int)(graph_z[idx_prev] * scale);
    int cy_z = GRAPH_Y + GRAPH_H/2 - (int)(graph_z[idx_curr] * scale);

    // Clamp dans la zone
    py_x = constrain(py_x, GRAPH_Y, GRAPH_Y + GRAPH_H);
    cy_x = constrain(cy_x, GRAPH_Y, GRAPH_Y + GRAPH_H);
    py_y = constrain(py_y, GRAPH_Y, GRAPH_Y + GRAPH_H);
    cy_y = constrain(cy_y, GRAPH_Y, GRAPH_Y + GRAPH_H);
    py_z = constrain(py_z, GRAPH_Y, GRAPH_Y + GRAPH_H);
    cy_z = constrain(cy_z, GRAPH_Y, GRAPH_Y + GRAPH_H);

    M5.Display.drawLine(px, py_x, cx, cy_x, TFT_RED);
    M5.Display.drawLine(px, py_y, cx, cy_y, COL_VERT);
    M5.Display.drawLine(px, py_z, cx, cy_z, COL_BLEU_CLAIR);
  }
}

// Met à jour le panneau historique (col 3)
void update_ui_historique() {
  int hx = COL3_X + 8;
  int hy = BODY_Y + 38;
  int hw = COL3_W - 16;

  // Effacer la zone
  M5.Display.fillRect(hx, hy, hw, BODY_H - 48, COL_PANEL);

  int entry_h = (BODY_H - 50) / HIST_MAX;

  for (int i = 0; i < hist_count; i++) {
    int idx = (hist_count - 1 - i); // plus récent en haut
    int ey  = hy + i * entry_h;

    uint32_t col = couleur_etat(historique[idx].etat);

    // Fond coloré subtil
    M5.Display.fillRoundRect(hx, ey + 2, hw, entry_h - 4, 4, COL_BG);
    M5.Display.drawFastVLine(hx + 3, ey + 2, entry_h - 4, col);

    // Heure
    M5.Display.setTextColor(COL_GRIS, COL_BG);
    M5.Display.setTextSize(1);
    M5.Display.setCursor(hx + 10, ey + 6);
    M5.Display.print(historique[idx].heure);

    // État
    M5.Display.setTextColor(col, COL_BG);
    M5.Display.setTextSize(2);
    M5.Display.setCursor(hx + 10, ey + 20);
    String e = historique[idx].etat;
    e.toUpperCase();
    M5.Display.print(e);

    // Confiance
    M5.Display.setTextColor(COL_GRIS, COL_BG);
    M5.Display.setTextSize(1);
    M5.Display.setCursor(hx + hw - 45, ey + 24);
    M5.Display.printf("%d%%", (int)(historique[idx].confiance * 100));
  }
}

// Écran d'alerte rouge clignotant
void show_alert_screen() {
  for (int blink = 0; blink < 3; blink++) {
    M5.Display.fillScreen(COL_ROUGE);
    M5.Display.setTextColor(COL_BLANC, COL_ROUGE);
    M5.Display.setTextSize(8);
    M5.Display.setTextDatum(MC_DATUM);
    M5.Display.drawString("CHUTE !", SCR_W/2, SCR_H/2 - 60);
    M5.Display.setTextSize(3);
    M5.Display.drawString("ALERTE ENVOYEE", SCR_W/2, SCR_H/2 + 40);
    M5.Display.drawString(format_time(), SCR_W/2, SCR_H/2 + 90);
    M5.Display.setTextDatum(TL_DATUM);
    M5.Speaker.tone(1000, 300);
    delay(400);
    M5.Display.fillScreen(COL_BG);
    delay(200);
  }
  redraw_full = true; // Redessin complet après l'alerte
}

// Affiche le message de feedback Node-RED
void show_feedback_screen(String msg) {
  M5.Display.fillRect(COL2_X + 5, BODY_Y + 5, COL2_W - 10, 80, 0x001F);
  M5.Display.drawRoundRect(COL2_X + 5, BODY_Y + 5, COL2_W - 10, 80, 6, COL_BLEU_CLAIR);
  M5.Display.setTextColor(COL_BLANC, 0x001F);
  M5.Display.setTextSize(1);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString("MESSAGE NODE-RED :", COL2_X + COL2_W/2, BODY_Y + 25);
  M5.Display.setTextSize(2);
  M5.Display.drawString(msg, COL2_X + COL2_W/2, BODY_Y + 55);
  M5.Display.setTextDatum(TL_DATUM);
}

// ==========================================
// 8. MQTT CALLBACK
// ==========================================
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.print("Message recu de Node-RED: ");
  Serial.println(message);

  M5.Speaker.tone(2000, 200);
  feedback_msg  = message;
  show_feedback = true;
  feedback_time = millis();
}

// ==========================================
// 9. WIFI & MQTT
// ==========================================
void setup_wifi() {
  M5.Display.setTextColor(COL_BLANC, COL_BG);
  M5.Display.setTextSize(2);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString("Connexion WiFi...", SCR_W/2, SCR_H/2);
  M5.Display.setTextDatum(TL_DATUM);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }

  espClient.setInsecure(); // Obligatoire pour le SSL HiveMQ
}

void reconnect_mqtt() {
  if (!client.connected()) {
    // Ajout de mqtt_user et mqtt_pass
    if (client.connect("M5Stack_Tab5_Senior1", mqtt_user, mqtt_pass)) {
      client.subscribe(topic_feedback);
    }
  }
}

// ==========================================
// 10. TFLITE
// ==========================================
bool setup_tflite() {
  tflite::InitializeTarget();
  tfl_model = tflite::GetModel(modele_chute_tflite);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) return false;

  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddUnidirectionalSequenceLSTM();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE
  );
  tfl_interpreter = &static_interpreter;

  if (tfl_interpreter->AllocateTensors() != kTfLiteOk) return false;

  input_tensor  = tfl_interpreter->input(0);
  output_tensor = tfl_interpreter->output(0);
  return true;
}

bool run_inference(float* flat_input, float* output) {
  for (int i = 0; i < NUMBER_OF_INPUTS; i++)
    input_tensor->data.f[i] = flat_input[i];
  if (tfl_interpreter->Invoke() != kTfLiteOk) return false;
  for (int i = 0; i < NUMBER_OF_OUTPUTS; i++)
    output[i] = output_tensor->data.f[i];
  return true;
}

// ==========================================
// 11. SETUP
// ==========================================
void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);

  M5.Display.setRotation(1); // Landscape
  M5.Display.setBrightness(200);

  Serial2.begin(115200, SERIAL_8N1, 16, 17);
  Serial.begin(115200);

  M5.Display.fillScreen(COL_BG);

  // Écran de démarrage
  M5.Display.setTextColor(COL_BLANC, COL_BG);
  M5.Display.setTextSize(3);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString("SYSTEME ANTI-CHUTE", SCR_W/2, SCR_H/2 - 60);
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(COL_GRIS, COL_BG);
  M5.Display.drawString("Initialisation...", SCR_W/2, SCR_H/2);
  M5.Display.setTextDatum(TL_DATUM);

  setup_wifi();

  client.setServer(mqtt_host, mqtt_port);
  client.setCallback(mqtt_callback);
  reconnect_mqtt();

  M5.Display.setTextColor(COL_VERT, COL_BG);
  M5.Display.setTextSize(2);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString("Chargement IA...", SCR_W/2, SCR_H/2 + 40);
  M5.Display.setTextDatum(TL_DATUM);

  bool ia_ok = setup_tflite();

  M5.Display.setTextColor(ia_ok ? COL_VERT : COL_ROUGE, COL_BG);
  M5.Display.setTextSize(2);
  M5.Display.setTextDatum(MC_DATUM);
  M5.Display.drawString(ia_ok ? "IA prete !" : "ERREUR IA", SCR_W/2, SCR_H/2 + 80);
  M5.Display.setTextDatum(TL_DATUM);

  delay(1500);

  // Dessiner l'interface principale
  draw_full_ui();
  update_ui_header();

  last_time_ms = millis();
}

// ==========================================
// 12. LOOP
// ==========================================
uint32_t last_header_update = 0;
uint32_t last_graph_update  = 0;

void loop() {
  M5.update();

  // Reconnexion MQTT si nécessaire
  if (!client.connected()) reconnect_mqtt();
  client.loop();

  // Mise à jour de l'heure
  update_time();

  // Mise à jour du header toutes les secondes
  if (millis() - last_header_update > 1000) {
    last_header_update = millis();
    update_ui_header();
  }

  // Redessiner l'interface complète si besoin (après alerte)
  if (redraw_full) {
    draw_full_ui();
    update_ui_header();
  }

  // Afficher le feedback Node-RED pendant 6 secondes
  if (show_feedback) {
    show_feedback_screen(feedback_msg);
    if (millis() - feedback_time > 6000) {
      show_feedback = false;
      // Effacer la zone feedback en redessinant le graphe
      M5.Display.fillRect(COL2_X + 5, BODY_Y + 5, COL2_W - 10, 80, COL_PANEL);
    }
  }

  // --- LECTURE DU MICRO:BIT ---
  if (Serial2.available()) {
    String jsonStr = Serial2.readStringUntil('\n');
    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, jsonStr);

    if (!error) {
      float ax = doc["acc_x"];
      float ay = doc["acc_y"];
      float az = doc["acc_z"];

      raw_buffer[buffer_index][0] = ax;
      raw_buffer[buffer_index][1] = ay;
      raw_buffer[buffer_index][2] = az;
      raw_buffer[buffer_index][3] = doc["magnitude"];
      buffer_index++;

      // Mise à jour du graphe accéléro toutes les 50ms
      if (millis() - last_graph_update > 50) {
        last_graph_update = millis();
        update_ui_graph(ax, ay, az);
      }

      // --- PRÉDICTION (fenêtre pleine) ---
      if (buffer_index >= WINDOW_SIZE) {

        // Normalisation + Flatten
        int k = 0;
        for (int i = 0; i < WINDOW_SIZE; i++)
          for (int j = 0; j < NUM_FEATURES; j++)
            flat_input_array[k++] = (raw_buffer[i][j] - scaler_mean[j]) / scaler_scale[j];

        if (run_inference(flat_input_array, output_array)) {

          // Trouver la classe max
          int   best_class = 0;
          float max_val    = output_array[0];
          for (int i = 1; i < NUMBER_OF_OUTPUTS; i++) {
            if (output_array[i] > max_val) {
              max_val    = output_array[i];
              best_class = i;
            }
          }
          etat_actuel        = classes[best_class];
          confiance_actuelle = max_val;

          // Télémétrie MQTT
          StaticJsonDocument<200> mqttDoc;
          mqttDoc["etat"]      = etat_actuel;
          mqttDoc["confiance"] = confiance_actuelle;
          char mqttBuffer[200];
          serializeJson(mqttDoc, mqttBuffer);
          client.publish(topic_telemetry, mqttBuffer);

          // Ajouter à l'historique
          if (hist_count < HIST_MAX) {
            historique[hist_count++] = {etat_actuel, confiance_actuelle, format_time()};
          } else {
            // Décaler et ajouter en fin
            for (int i = 0; i < HIST_MAX - 1; i++)
              historique[i] = historique[i + 1];
            historique[HIST_MAX - 1] = {etat_actuel, confiance_actuelle, format_time()};
          }

          // Mettre à jour l'UI état et historique
          update_ui_etat();
          update_ui_historique();

          // Alerte chute
          if (etat_actuel == "chute" && confiance_actuelle > 0.85) {
            client.publish(topic_alerte, "ALERTE: CHUTE CONFIRMEE !");
            show_alert_screen();
          }
        }

        // Fenêtre glissante 50%
        const int STEP_SIZE = 50;
        for (int i = 0; i < WINDOW_SIZE - STEP_SIZE; i++)
          for (int j = 0; j < NUM_FEATURES; j++)
            raw_buffer[i][j] = raw_buffer[i + STEP_SIZE][j];
        buffer_index = WINDOW_SIZE - STEP_SIZE;
      }
    }
  }
}