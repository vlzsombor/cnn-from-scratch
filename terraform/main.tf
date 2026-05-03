terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}
provider "azurerm" {
  features {}
}

variable "volume_path" {
  type        = string
  description = "GitHub Username"
}
variable "github_image" {
  type        = string
  description = "GitHub Username"
}
variable "github_username" {
  type        = string
  description = "GitHub Username"
}

variable "github_token" {
  type        = string
  sensitive   = true
  description = "GitHub Personal Access Token"
}
variable "resource_group_name" {
  type    = string
  default = "myTFResourceGroup"
}

output "resource_group_name" {
  value = var.resource_group_name
}
variable "location" {
  type    = string
  default = "germanywestcentral"
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_storage_account" "storage" {
  name                     = "stcheap${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  access_tier              = "Cool"

  blob_properties {
    versioning_enabled = false
  }
}

resource "azurerm_storage_container" "container" {
  name                  = "data"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

resource "azurerm_storage_share" "share" {
  name                 = "fileshare"
  storage_account_name = azurerm_storage_account.storage.name
  quota                = 1
}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

output "storage_account_name" {
  value = azurerm_storage_account.storage.name
}
output "log_command" {
  value = "az container logs --resource-group ${azurerm_resource_group.rg.name} --name ${azurerm_container_group.container.name} --follow"
}

resource "azurerm_container_group" "container" {
  name                = "aci-app"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  os_type             = "Linux"
  restart_policy      = "Never"
  ip_address_type = "None"
  container {
    name   = "app"
    image  = var.github_image
    cpu    = "0.5"
    memory = "1.5"

    volume {
      name                 = "data"
      mount_path           = var.volume_path
      storage_account_name = azurerm_storage_account.storage.name
      storage_account_key  = azurerm_storage_account.storage.primary_access_key
      share_name           = azurerm_storage_share.share.name
    }
  }

  image_registry_credential {
    server   = "ghcr.io"
    username = var.github_username
    password = var.github_token
  }
}


resource "azurerm_container_group" "container-better" {
  name                = "aci-app-better"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  os_type             = "Linux"
  restart_policy      = "Never"
  ip_address_type = "None"
  container {
    name   = "app"
    image  = var.github_image
    cpu    = "2"
    memory = "4"

    volume {
      name                 = "data"
      mount_path           = var.volume_path
      storage_account_name = azurerm_storage_account.storage.name
      storage_account_key  = azurerm_storage_account.storage.primary_access_key
      share_name           = azurerm_storage_share.share.name
    }
  }

  image_registry_credential {
    server   = "ghcr.io"
    username = var.github_username
    password = var.github_token
  }
}
output "container_fqdn" {
  value = azurerm_container_group.container.fqdn
}

