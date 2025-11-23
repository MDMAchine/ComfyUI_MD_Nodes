# ✅ SETUP CHECKLIST - ACE-Step LoRA Training Suite v2.1

## 📋 File Placement Verification

Copy this checklist and mark items as you place them:

### **Root Level Files** (`D:\Ace-Step_Fork\`)
- [ ] lora_trainer_automation_v2.1_FINAL.py
- [ ] README_MASTER.md
- [ ] DEPLOYMENT_GUIDE.md
- [ ] FINAL_PACKAGE_SUMMARY.md
- [ ] LoRA_Training_Quick_Reference.md
- [ ] Troubleshooting_Guide.md
- [ ] ACE_Step_LoRA_Training_Guide.docx
- [ ] Changelog_and_Fixes.md

### **Tools Directory** (`D:\Ace-Step_Fork\Tools\`)
- [ ] add_tag.py
- [ ] lora_magician_pro.py
- [ ] normalize_audio.py

### **ACE-Step Directory** (`D:\Ace-Step_Fork\ACE-Step\`)
- [ ] trainer_new.py (YOUR modified version)
- [ ] generate_prompts_lyrics.py (YOUR modified version)
- [ ] Scripts\python.exe (virtual environment)
- [ ] config\lora_config_transformer_only.json
- [ ] (All other ACE-Step files)

---

## 🧪 Quick Test

### **Test 1: Check Python**
```bash
D:\Ace-Step_Fork\ACE-Step\Scripts\python.exe --version
```
Expected: `Python 3.10.x` or `Python 3.11.x`

### **Test 2: Check Tools**
```bash
dir D:\Ace-Step_Fork\Tools\*.py
```
Expected: Should show 3 files (add_tag.py, lora_magician_pro.py, normalize_audio.py)

### **Test 3: Check Automation Script**
```bash
dir D:\Ace-Step_Fork\lora_trainer_automation_v2.1_FINAL.py
```
Expected: Should show the file

### **Test 4: Check Documentation**
```bash
dir D:\Ace-Step_Fork\*.md
```
Expected: Should show 6 markdown files

### **Test 5: Activate Environment**
```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat
```
Expected: Prompt changes to `(ACE-Step) D:\Ace-Step_Fork\ACE-Step>`

### **Test 6: Import Check**
```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```
Expected: `True` (if you have NVIDIA GPU)

---

## 🚀 First Run Test

```bash
# 1. Navigate to root
cd D:\Ace-Step_Fork

# 2. Run automation (will prompt for config)
python lora_trainer_automation_v2.1_FINAL.py

# 3. When prompted:
#    - Accept default Ace-Step_Fork path
#    - Enter your audio directory
#    - Choose step 1
#    - Configure with small values for testing:
#      * Max steps: 100
#      * Workers: 2
```

This will verify the automation script runs correctly without a full training session.

---

## 📁 Directory Structure Verification

Run this to check your structure:
```bash
tree D:\Ace-Step_Fork /F /A > structure.txt
```

Your structure should match:
```
D:\Ace-Step_Fork\
├── *.md (6 documentation files)
├── *.docx (1 Word document)
├── lora_trainer_automation_v2.1_FINAL.py
├── Tools\
│   ├── add_tag.py
│   ├── lora_magician_pro.py
│   └── normalize_audio.py
├── ACE-Step\
│   ├── Scripts\python.exe
│   ├── trainer_new.py
│   ├── generate_prompts_lyrics.py
│   ├── config\
│   └── (other ACE-Step files)
├── processed_data\ (created during training)
├── runs\ (created during training)
└── LoRa\ (optional, for final models)
```

---

## ✅ Ready to Train Checklist

Before your first real training session:

### **Environment**
- [ ] Virtual environment activates successfully
- [ ] PyTorch with CUDA support installed
- [ ] All requirements from `requirements.txt` installed
- [ ] GPU detected (`nvidia-smi` works)

### **Data**
- [ ] Audio files are organized in one folder
- [ ] Audio files are normalized (if using normalize_audio.py)
- [ ] At least 20 audio files (more is better)
- [ ] Files are high quality (192kbps+ for MP3)

### **Disk Space**
- [ ] At least 50GB free space on drive
- [ ] Fast storage (SSD recommended)
- [ ] Enough space for checkpoints (plan ~5GB per 5000 steps)

### **Tools**
- [ ] add_tag.py works: `python Tools\add_tag.py --help`
- [ ] lora_magician_pro.py works: `python Tools\lora_magician_pro.py --help`
- [ ] normalize_audio.py works (if needed)

### **Documentation**
- [ ] Read README_MASTER.md
- [ ] Bookmarked DEPLOYMENT_GUIDE.md
- [ ] Know where to find Quick Reference
- [ ] Know where to find Troubleshooting Guide

---

## 🎯 Quick Start Command

Once everything is verified:

```bash
cd D:\Ace-Step_Fork
.\ACE-Step\Scripts\activate.bat
python lora_trainer_automation_v2.1_FINAL.py
```

Follow the prompts and train!

---

## 📞 If Something's Wrong

### **Script won't start:**
1. Check Python path: `D:\Ace-Step_Fork\ACE-Step\Scripts\python.exe`
2. Check file exists: `D:\Ace-Step_Fork\lora_trainer_automation_v2.1_FINAL.py`
3. Check you're in the right directory: `cd D:\Ace-Step_Fork`

### **Can't find tools:**
1. Check Tools directory exists: `dir D:\Ace-Step_Fork\Tools`
2. Check files are there: `dir D:\Ace-Step_Fork\Tools\*.py`
3. Check paths in error message match expected paths

### **Import errors:**
```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat
pip install -r requirements.txt
```

### **GPU not detected:**
```bash
nvidia-smi
# Should show your GPU

python -c "import torch; print(torch.cuda.is_available())"
# Should print True
```

---

## 📊 Validation Commands

Run these to verify everything:

```bash
# 1. Check directory structure
cd D:\Ace-Step_Fork
dir

# 2. Check Tools
dir Tools

# 3. Check ACE-Step
dir ACE-Step

# 4. Check Python
ACE-Step\Scripts\python.exe --version

# 5. Check GPU
nvidia-smi

# 6. Check PyTorch + CUDA
ACE-Step\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 7. Test automation script (dry run)
ACE-Step\Scripts\python.exe lora_trainer_automation_v2.1_FINAL.py
```

---

## 🎉 Success Indicators

You're ready when:
✅ All checkboxes above are marked  
✅ All test commands run without errors  
✅ Documentation is accessible  
✅ GPU is detected  
✅ Virtual environment activates  
✅ Automation script starts  

---

## 📝 Notes

- **Save this checklist** - useful for troubleshooting later
- **Run tests after any updates** to the environment
- **Keep documentation handy** during first training session
- **Start with small max_steps** (100-500) to test the pipeline

---

**Your setup is complete when all items are checked! 🚀**