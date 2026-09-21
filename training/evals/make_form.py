"""Generate a real AcroForm PDF, so pdf_fill_form has something to fill.

Fillable forms are oddly hard to find for free, and the tool only works on
AcroForms -- pdf_stamp draws on a page but cannot add fields, so nothing in the
app can produce one. Authoring it here is easier than hunting.

Deliberately includes every field type the tool's schema mentions:
  text      plain values
  checkbox  true/false
  choice    one of a listed set
  radio     one of a listed set, grouped

A form with only text fields would exercise a quarter of the tool.
"""
import sys

from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

OUT = sys.argv[1] if len(sys.argv) > 1 else "equipment-request.pdf"
W, H = LETTER

c = canvas.Canvas(OUT, pagesize=LETTER)
c.setTitle("Equipment Request Form")

c.setFont("Helvetica-Bold", 18)
c.drawString(60, H - 70, "Equipment Request Form")
c.setFont("Helvetica", 9)
c.drawString(60, H - 88, "Internal use. Complete all required fields.")
c.line(60, H - 96, W - 60, H - 96)

form = c.acroForm
y = H - 140


def label(text, yy):
    c.setFont("Helvetica", 10)
    c.drawString(60, yy + 4, text)


# --- text fields
for name, tip in [
    ("employee_name", "Full name"),
    ("employee_id", "Employee ID"),
    ("email", "Work email"),
    ("department", "Department"),
]:
    label(tip, y)
    form.textfield(name=name, tooltip=tip, x=200, y=y - 4, width=300, height=20,
                   borderColor=black, fillColor=white, textColor=black,
                   forceBorder=True)
    y -= 34

# --- choice (dropdown)
label("Equipment type", y)
form.choice(name="equipment_type", tooltip="Equipment type",
            value="Laptop", options=["Laptop", "Monitor", "Keyboard",
                                     "Docking station", "Headset"],
            x=200, y=y - 4, width=300, height=20,
            borderColor=black, fillColor=white, textColor=black, forceBorder=True)
y -= 34

# --- radio group
label("Urgency", y)
for i, opt in enumerate(["Standard", "Expedited", "Critical"]):
    form.radio(name="urgency", tooltip="Urgency", value=opt,
               selected=(opt == "Standard"),
               x=200 + i * 110, y=y - 4, size=16,
               borderColor=black, fillColor=white, textColor=black,
               forceBorder=True)
    c.setFont("Helvetica", 9)
    c.drawString(220 + i * 110, y + 1, opt)
y -= 34

# --- checkboxes
label("Accessories", y)
for i, opt in enumerate(["carry_case", "extra_cable", "warranty"]):
    form.checkbox(name=opt, tooltip=opt.replace("_", " "), checked=False,
                  x=200 + i * 130, y=y - 4, size=16,
                  borderColor=black, fillColor=white, textColor=black,
                  forceBorder=True)
    c.setFont("Helvetica", 9)
    c.drawString(220 + i * 130, y + 1, opt.replace("_", " "))
y -= 40

# --- multiline
label("Justification", y - 40)
form.textfield(name="justification", tooltip="Justification",
               x=200, y=y - 60, width=300, height=60, fieldFlags="multiline",
               borderColor=black, fillColor=white, textColor=black,
               forceBorder=True)

c.setFont("Helvetica-Oblique", 8)
c.drawString(60, 60, "Form generated for testing. Not a real request process.")
c.save()
print("wrote", OUT)
