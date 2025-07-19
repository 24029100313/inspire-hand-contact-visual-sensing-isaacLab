#!/usr/bin/env python3
"""
Contact Sensor Diagnostic Tool for Isaac Sim
Complete Contact Sensor configuration diagnostic and verification tool
"""

import omni.usd
from pxr import UsdPhysics, PhysxSchema, UsdGeom, Sdf
import carb


class ContactSensorDiagnostic:
    """Contact Sensor diagnostic tool class"""
    
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.issues = []
        self.warnings = []
        
    def print_status(self, message, status="INFO"):
        """Print status information"""
        if status == "ERROR":
            print(f"❌ {message}")
            self.issues.append(message)
        elif status == "WARNING":
            print(f"⚠️  {message}")
            self.warnings.append(message)
        elif status == "SUCCESS":
            print(f"✅ {message}")
        else:
            print(f"ℹ️  {message}")
    
    def check_prim_exists(self, prim_path):
        """Check if Prim exists"""
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            self.print_status(f"Prim does not exist: {prim_path}", "ERROR")
            return False
        self.print_status(f"Prim exists: {prim_path}", "SUCCESS")
        return True
    
    def check_contact_sensor_api(self, sensor_path):
        """Check Contact Sensor API"""
        if not self.check_prim_exists(sensor_path):
            return False
            
        sensor_prim = self.stage.GetPrimAtPath(sensor_path)
        
        # Check if it has Contact Report API
        if not sensor_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            self.print_status(f"Missing PhysxContactReportAPI: {sensor_path}", "ERROR")
            return False
        
        self.print_status(f"PhysxContactReportAPI exists: {sensor_path}", "SUCCESS")
        return True
    
    def check_sensor_parameters(self, sensor_path):
        """Check sensor parameter configuration"""
        if not self.check_contact_sensor_api(sensor_path):
            return False
            
        sensor_prim = self.stage.GetPrimAtPath(sensor_path)
        contact_api = PhysxSchema.PhysxContactReportAPI(sensor_prim)
        
        # Check key parameters
        enabled = contact_api.GetEnabledAttr().Get()
        period = contact_api.GetSensorPeriodAttr().Get()
        radius = contact_api.GetRadiusAttr().Get()
        min_threshold = contact_api.GetMinThresholdAttr().Get()
        max_threshold = contact_api.GetMaxThresholdAttr().Get()
        
        print(f"\n📊 Sensor parameters - {sensor_path}:")
        print(f"   Enabled: {enabled}")
        print(f"   Period: {period}")
        print(f"   Radius: {radius}")
        print(f"   Min threshold: {min_threshold}")
        print(f"   Max threshold: {max_threshold}")
        
        # Validate parameter reasonableness
        if not enabled:
            self.print_status("Sensor not enabled", "ERROR")
            return False
        else:
            self.print_status("Sensor enabled", "SUCCESS")
        
        if period is None or period <= 0:
            self.print_status("Sensor period not set or invalid", "ERROR")
            return False
        elif period > 0.1:
            self.print_status(f"Sensor period too large ({period}s), recommend 0.02s", "WARNING")
        else:
            self.print_status(f"Sensor period reasonable ({period}s)", "SUCCESS")
        
        if radius is None or radius <= 0:
            self.print_status("Sensor radius not set or invalid", "ERROR")
            return False
        elif radius < 0.005:
            self.print_status(f"Sensor radius too small ({radius}m), may not detect contact", "WARNING")
        elif radius > 0.1:
            self.print_status(f"Sensor radius too large ({radius}m), may cause false positives", "WARNING")
        else:
            self.print_status(f"Sensor radius reasonable ({radius}m)", "SUCCESS")
        
        return True
    
    def check_parent_collider(self, sensor_path):
        """Check parent object collision configuration"""
        sensor_prim = self.stage.GetPrimAtPath(sensor_path)
        
        # First check if sensor_prim is valid
        if not sensor_prim.IsValid():
            self.print_status(f"Sensor Prim invalid: {sensor_path}", "ERROR")
            return False
        
        parent_prim = sensor_prim.GetParent()
        
        if not parent_prim.IsValid():
            self.print_status(f"Sensor parent object invalid: {sensor_path}", "ERROR")
            return False
        
        parent_path = parent_prim.GetPath()
        self.print_status(f"Checking parent collision: {parent_path}", "INFO")
        
        # Check CollisionAPI
        if not parent_prim.HasAPI(UsdPhysics.CollisionAPI):
            self.print_status(f"Parent object missing CollisionAPI: {parent_path}", "ERROR")
            return False
        
        collision_api = UsdPhysics.CollisionAPI(parent_prim)
        collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
        
        if not collision_enabled:
            self.print_status(f"Parent object collision not enabled: {parent_path}", "ERROR")
            return False
        
        self.print_status(f"Parent object collision configured correctly: {parent_path}", "SUCCESS")
        return True
    
    def check_physics_scene(self):
        """Check physics scene"""
        physics_scene_found = False
        
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "PhysicsScene":
                physics_scene_found = True
                physics_scene_path = prim.GetPath()
                self.print_status(f"Found physics scene: {physics_scene_path}", "SUCCESS")
                break
        
        if not physics_scene_found:
            self.print_status("No physics scene found, need to create PhysicsScene", "ERROR")
            return False
        
        return True
    
    def check_simulation_running(self):
        """Check if simulation is running"""
        try:
            timeline = omni.timeline.get_timeline_interface()
            is_playing = timeline.is_playing()
            
            if not is_playing:
                self.print_status("Simulation not running, need to click Play button", "WARNING")
                return False
            else:
                self.print_status("Simulation running", "SUCCESS")
                return True
        except:
            self.print_status("Unable to detect simulation state", "WARNING")
            return False
    
    def auto_fix_sensor(self, sensor_path):
        """Auto fix sensor configuration"""
        self.print_status(f"Attempting to auto-fix sensor: {sensor_path}", "INFO")
        
        if not self.check_prim_exists(sensor_path):
            # If sensor doesn't exist, try to create it
            parent_path = str(Sdf.Path(sensor_path).GetParentPath())
            sensor_name = str(Sdf.Path(sensor_path).name)
            return self.create_contact_sensor(parent_path, sensor_name)
        
        sensor_prim = self.stage.GetPrimAtPath(sensor_path)
        
        # Add Contact Report API (if not exists)
        if not sensor_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            contact_api = PhysxSchema.PhysxContactReportAPI.Apply(sensor_prim)
            self.print_status("Added PhysxContactReportAPI", "SUCCESS")
        else:
            contact_api = PhysxSchema.PhysxContactReportAPI(sensor_prim)
        
        # Set recommended parameters
        contact_api.GetEnabledAttr().Set(True)
        contact_api.GetSensorPeriodAttr().Set(0.02)  # 50Hz
        contact_api.GetRadiusAttr().Set(0.02)        # 2cm
        contact_api.GetMinThresholdAttr().Set(0.0)
        contact_api.GetMaxThresholdAttr().Set(1000000.0)
        
        self.print_status(f"Sensor configuration fixed: {sensor_path}", "SUCCESS")
        return True
    
    def create_contact_sensor(self, parent_path, sensor_name="Contact_Sensor"):
        """Create new Contact Sensor"""
        if not self.check_prim_exists(parent_path):
            self.print_status(f"Parent object does not exist: {parent_path}", "ERROR")
            return False
        
        sensor_path = f"{parent_path}/{sensor_name}"
        sensor_prim = self.stage.DefinePrim(sensor_path, "Scope")
        
        # Add Contact Report API
        contact_api = PhysxSchema.PhysxContactReportAPI.Apply(sensor_prim)
        
        # Set recommended parameters
        contact_api.GetEnabledAttr().Set(True)
        contact_api.GetSensorPeriodAttr().Set(0.02)  # 50Hz
        contact_api.GetRadiusAttr().Set(0.02)        # 2cm
        contact_api.GetMinThresholdAttr().Set(0.0)
        contact_api.GetMaxThresholdAttr().Set(1000000.0)
        
        self.print_status(f"Created new Contact Sensor: {sensor_path}", "SUCCESS")
        return True
    
    def find_all_sensors(self):
        """Find all existing contact sensors"""
        sensors = []
        for prim in self.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                sensors.append(str(prim.GetPath()))
        return sensors
    
    def comprehensive_check(self, sensor_paths):
        """Comprehensive check of multiple sensors"""
        print("🔍 Starting comprehensive Contact Sensor configuration diagnosis...")
        print("=" * 60)
        
        self.issues = []
        self.warnings = []
        
        # First find all existing sensors
        existing_sensors = self.find_all_sensors()
        if existing_sensors:
            print(f"📍 Found existing sensors: {len(existing_sensors)}")
            for sensor in existing_sensors:
                print(f"   - {sensor}")
        else:
            print("📍 No existing sensors found")
        
        # Check physics scene
        self.check_physics_scene()
        
        # Check simulation state
        self.check_simulation_running()
        
        # Check each sensor
        for i, sensor_path in enumerate(sensor_paths):
            print(f"\n📍 Checking sensor {i+1}/{len(sensor_paths)}: {sensor_path}")
            print("-" * 50)
            
            # Check sensor parameters
            sensor_params_ok = self.check_sensor_parameters(sensor_path)
            
            # Only check parent collision if sensor exists
            if sensor_params_ok:
                # Check parent collision
                self.check_parent_collider(sensor_path)
            else:
                print(f"⚠️  Skipping parent collision check (sensor doesn't exist or misconfigured): {sensor_path}")
        
        # Summary report
        print("\n" + "=" * 60)
        print("📋 Diagnosis summary:")
        print(f"⚠️  Warning items: {len(self.warnings)}")
        print(f"❌ Error items: {len(self.issues)}")
        
        if self.issues:
            print("\n🔴 Errors that need fixing:")
            for issue in self.issues:
                print(f"   - {issue}")
        
        if self.warnings:
            print("\n🟡 Warnings to note:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.issues and not self.warnings:
            print("\n🎉 All sensors configured perfectly!")
            return True
        else:
            print(f"\n🛠️  Can run diagnostic.auto_fix_all() for auto-repair")
            return False
    
    def auto_fix_all(self, sensor_paths):
        """Auto fix all sensors"""
        print("🔧 Starting auto-fix for all sensors...")
        
        success_count = 0
        for sensor_path in sensor_paths:
            if self.auto_fix_sensor(sensor_path):
                success_count += 1
        
        print(f"\nFix completed: {success_count}/{len(sensor_paths)} sensors")
        
        # Re-check
        print("\n🔍 Re-verifying fix results...")
        return self.comprehensive_check(sensor_paths)


def main():
    """Main function - usage example"""
    
    # Create diagnostic tool
    diagnostic = ContactSensorDiagnostic()
    
    # Define sensor paths to check - modify according to your setup
    sensor_paths = [
        "/World/envs/env_0000/Robot/panda_leftfinger/panda_leftfinger/Contact_Sensor",
        "/World/envs/env_0000/Robot/panda_rightfinger/panda_rightfinger/Contact_Sensor"
    ]
    
    # Execute comprehensive check
    is_all_good = diagnostic.comprehensive_check(sensor_paths)
    
    if not is_all_good:
        # Auto fix
        print("\n🔧 Auto-fixing detected issues...")
        diagnostic.auto_fix_all(sensor_paths)
    
    return diagnostic


# Usage example
if __name__ == "__main__":
    # Run diagnosis directly
    diagnostic = main()
    
    # Or create and use manually
    # diagnostic = ContactSensorDiagnostic()
    # sensor_paths = ["/World/Robot/panda_leftfinger/panda_leftfinger/Contact_Sensor"]
    # diagnostic.comprehensive_check(sensor_paths) 